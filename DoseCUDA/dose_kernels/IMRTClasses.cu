#include "./IMRTClasses.cuh"
#include "MemoryClasses.h"
#include <memory>


__host__ __device__ float interp(float x, const float * xd, const float * yd, const int n){

	int i = lowerBound(xd, n, x);

	if (i == n) {
		return yd[i - 1];
	} else if (i == 0) {
		return yd[i];
	} else {
		auto factor = (x - xd[i - 1]) / (xd[i] - xd[i - 1]);
		return fmaf(yd[i] - yd[i - 1], factor, yd[i - 1]);
	}
}

__host__ IMRTBeam::IMRTBeam(IMRTBeam * h_beam) : CudaBeam(h_beam) {

	this->model = h_beam->model;
	this->collimator_angle = h_beam->collimator_angle;
	this->mu = h_beam->mu;
	this->sinca = h_beam->sinca;
	this->cosca = h_beam->cosca;
	this->n_mlc_pairs = h_beam->n_mlc_pairs;

}

__host__ IMRTBeam::IMRTBeam(float * iso, float gantry_angle, float couch_angle, float collimator_angle, const Model * model)
		: CudaBeam(iso, gantry_angle, couch_angle, model->primary_src_dist) {

	this->model = *model;
	this->collimator_angle = collimator_angle;

	float ca = collimator_angle * CUDART_PI_F / 180.0f;
	this->sinca = sin(ca);
	this->cosca = cos(ca);
}

__device__ void IMRTBeam::pointXYZImageToHead(const PointXYZ * point_img, PointXYZ * point_head){

	float sinx, cosx;

	//table rotation - rotate about y-axis
	float xt, yt, zt;
	sinx = -this->sinta;
	cosx = this->costa;
	xt = point_img->x * cosx + point_img->z * sinx;
	yt = point_img->y;
	zt = -point_img->x * sinx + point_img->z * cosx;

	//gantry rotation - rotate about z-axis
	float xg, yg, zg;
	sinx = -this->singa;
	cosx = this->cosga;
	xg = xt * cosx - yt * sinx;
	yg = xt * sinx + yt * cosx;
	zg = zt;

	//collimator rotation = rotate about y-axis
	float xc, yc, zc;
	sinx = -this->sinca;
	cosx = this->cosca;
	xc  = xg * cosx + zg * sinx;
	yc = yg;
	zc = -xg * sinx + zg * cosx;


	//swap final coordinates to match DICOM nozzle coordinate system
	//for an AP beam:
	//	beam travels in negative z direction
	//	positive x is to the patient's left
	//	positive y is to the patient's superior
	point_head->x = -xc;
	point_head->y = zc;
	point_head->z = yc;

}

__device__ void IMRTBeam::pointXYZHeadToImage(const PointXYZ * point_head, PointXYZ * point_img){

	float sinx, cosx;

	//convert back to DICOM patient LPS coordinates
	float xz = -point_head->x;
	float yz = point_head->z;
	float zz = point_head->y;

	//collimator rotation = rotate about y-axis (again, negative direction)
	float xc, yc, zc;
	sinx = this->sinca;
	cosx = this->cosca;
	xc  = xz * cosx + zz * sinx;
	yc = yz;
	zc = -xz * sinx + zz * cosx;

	//gantry rotation - rotate about z-axis (negative direction)
	float xg, yg, zg;
	sinx = this->singa;
	cosx = this->cosga;
	xg = xc * cosx - yc * sinx;
	yg = xc * sinx + yc * cosx;
	zg = zc;

	//table rotation - rotate about y-axis (negative direction)
	float xt, yt, zt;
	sinx = this->sinta;
	cosx = this->costa;
	xt = xg * cosx + zg * sinx;
	yt = yg;
	zt = -xg * sinx + zg * cosx;

	point_img->x = xt;
	point_img->y = yt;
	point_img->z = zt;

}

__device__ float IMRTBeam::headTransmission(const PointXYZ* point_xyz, const float iso_to_source, const float source_sigma){

	const float mlc_scale = (model.primary_src_dist - model.mlc_distance) / model.primary_src_dist;
	const float divergence_scale = (iso_to_source - point_xyz->z) / (model.mlc_distance - point_xyz->z);
	const float invSqrt2_x = 1.0f / (source_sigma * sqrtf(2.f));
    const float invSqrt2_y = 1.0f / (source_sigma * sqrtf(2.f));
	
	// ========== ETAPA 1: JAWS (retângulo divergente) ==========
	float jaw_factor = 1.0f;
	float jaw_open = 1.0f;
	if (model.has_xjaws || model.has_yjaws) {
		// Projetar jaws para plano do voxel
		const float xLeft_jaw  = ((xjaws[0] * mlc_scale - point_xyz->x) * divergence_scale) + point_xyz->x;
		const float xRight_jaw = ((xjaws[1] * mlc_scale - point_xyz->x) * divergence_scale) + point_xyz->x;
		const float yBot_jaw   = ((yjaws[0] * mlc_scale - point_xyz->y) * divergence_scale) + point_xyz->y;
		const float yTop_jaw   = ((yjaws[1] * mlc_scale - point_xyz->y) * divergence_scale) + point_xyz->y;

		const float exposedSourceX_jaw = 0.5f * (erff(xRight_jaw * invSqrt2_x) - erff(xLeft_jaw * invSqrt2_x));
		const float exposedSourceY_jaw = 0.5f * (erff(yTop_jaw * invSqrt2_y) - erff(yBot_jaw * invSqrt2_y));

		jaw_open = fmaxf(0.0f, exposedSourceX_jaw * exposedSourceY_jaw);
		// Combine open fraction with jaw transmission (leakage when closed)
		jaw_factor = jaw_open + (1.0f - jaw_open) * model.jaw_transmission;
	}

	// ========== ETAPA 2: MLC ==========
	float leaf_open = 0.0f;
	bool leaf_found = false;

	for (int i = 0; i < this->n_mlc_pairs; ++i) {

		const float yBottom = (this->mlc[i].y_offset - 0.5f * this->mlc[i].y_width) * mlc_scale;
		const float yTop    = (this->mlc[i].y_offset + 0.5f * this->mlc[i].y_width) * mlc_scale;

		// Select only the leaf pair that covers this Y position
		if (point_xyz->y < yBottom || point_xyz->y > yTop) {
			continue;
		}
		leaf_found = true;

		const float xLeft  = (this->mlc[i].x1 * mlc_scale);
		const float xRight = (this->mlc[i].x2 * mlc_scale);

		const float tipMLC1 = ((xLeft - point_xyz->x) * divergence_scale) + point_xyz->x;
		const float tipMLC2 = ((xRight - point_xyz->x) * divergence_scale) + point_xyz->x;

		const float edgeMLC1 = ((yBottom - point_xyz->y) * divergence_scale) + point_xyz->y;
		const float edgeMLC2 = ((yTop - point_xyz->y) * divergence_scale) + point_xyz->y;

		const float exposedSourceX = 0.5f * (erff(tipMLC2 * invSqrt2_x) - erff(tipMLC1 * invSqrt2_x));
		const float exposedSourceY = 0.5f * (erff(edgeMLC2 * invSqrt2_y) - erff(edgeMLC1 * invSqrt2_y));

		leaf_open = fmaxf(0.0f, exposedSourceX * exposedSourceY);
		break;
	}

	// If the voxel is outside any leaf pair, treat as blocked by MLC housing
	if (!leaf_found) {
		leaf_open = 0.0f;
	}

	float mlc_factor = leaf_open + (1.0f - leaf_open) * model.mlc_transmission;

	// ETAPA 3: COMBINAÇÃO (jaws ∩ MLC)
	return jaw_factor * mlc_factor;

}

__device__ void IMRTBeam::offAxisFactors(const PointXYZ * point_xyz, float * off_axis_factor, float * off_axis_softening){

	const float distance_to_source = model.primary_src_dist - point_xyz->z;
	const float distance_to_cax = hypotf(point_xyz->x, point_xyz->y) * distance_to_source / model.primary_src_dist;

	int i = lowerBound(this->model.profile_radius, this->model.n_profile_points, distance_to_cax);

	if (i == this->model.n_profile_points) {
		*off_axis_factor = this->model.profile_intensities[i - 1];
		*off_axis_softening = this->model.profile_softening[i - 1];
	} else if (i == 0) {
		*off_axis_factor = this->model.profile_intensities[i];
		*off_axis_softening = this->model.profile_softening[i];
	} else {
		auto mult = (distance_to_cax - this->model.profile_radius[i - 1]) / (this->model.profile_radius[i] - this->model.profile_radius[i - 1]);
		*off_axis_factor = fmaf(this->model.profile_intensities[i] - this->model.profile_intensities[i - 1], mult, this->model.profile_intensities[i - 1]);
		*off_axis_softening = fmaf(this->model.profile_softening[i] - this->model.profile_softening[i - 1], mult, this->model.profile_softening[i - 1]);
	}

}

__device__ void IMRTBeam::kernelTilt(const PointXYZ * vox_img_xyz, PointXYZ * vec_img) {

	PointXYZ uvec;
	this->unitVectorToSource(vox_img_xyz, &uvec);

	/* Compute z in the img tangent space as the unit vector from iso to source */
	PointXYZ axis{ };
	this->unitVectorToSource(&axis, &axis);

	/* Cosine of the tilt angle θ */
	auto costh = xyz_dotproduct(axis, uvec);

	/* Tilt axis, scaled by sin(θ) */
	axis = xyz_crossproduct(axis, uvec);

	/* Rodrigues' formula (second term first) */
	auto result = xyz_crossproduct(axis, *vec_img);

	/* First term */
	result.x += vec_img->x * costh;
	result.y += vec_img->y * costh;
	result.z += vec_img->z * costh;

	/* Last term */
	auto scal = xyz_dotproduct(axis, *vec_img) / (1.0f + costh);
	result.x += axis.x * scal;
	result.y += axis.y * scal;
	result.z += axis.z * scal;

	*vec_img = result;
}


__host__ IMRTDose::IMRTDose(CudaDose * h_dose) : CudaDose(h_dose) {}


__global__ void termaKernel(IMRTDose * dose, IMRTBeam * beam, float * TERMAArray, float * ElectronArray){

	PointIJK vox_ijk;
	vox_ijk.k = threadIdx.x + (blockIdx.x * blockDim.x);
	vox_ijk.j = threadIdx.y + (blockIdx.y * blockDim.y);
	vox_ijk.i = threadIdx.z + (blockIdx.z * blockDim.z);

	if(!dose->pointIJKWithinImage(&vox_ijk)) {
		return;
	}

	unsigned vox_index = dose->pointIJKtoIndex(&vox_ijk);

	PointXYZ vox_xyz, vox_head_xyz;
	dose->pointIJKtoXYZ(&vox_ijk, &vox_xyz, beam);
	beam->pointXYZImageToHead(&vox_xyz, &vox_head_xyz);

	float distance_to_primary_source = beam->model.primary_src_dist - vox_head_xyz.z;
	float distance_to_scatter_source = beam->model.scatter_src_dist - vox_head_xyz.z;
	float off_axis_factor, off_axis_softening;
	beam->offAxisFactors(&vox_head_xyz, &off_axis_factor, &off_axis_softening);
	float primary_transmission = beam->headTransmission(&vox_head_xyz, beam->model.primary_src_dist, beam->model.primary_src_size);
	float scatter_transmission = beam->headTransmission(&vox_head_xyz, beam->model.scatter_src_dist, beam->model.scatter_src_size);
	float wet = dose->WETArray[vox_index];
	float electron = 0.f;

	// Compute primary and scatter TERMA contributions separately (AAA-like)
	float terma_primary = 0.0f;
	float terma_scatter = 0.0f;

	for(int i = 0; i < beam->model.n_spectral_energies; i++){
		float mu = beam->model.spectrum_attenuation_coefficients[i];
		float wP = beam->model.spectrum_primary_weights[i];
		float wS = beam->model.spectrum_scatter_weights[i];

		float atten = expf(-mu * wet * off_axis_softening);

		// Primary source term: inverse-square and primary weight
		float prim_term = wP * atten * sqr(beam->model.primary_src_dist / distance_to_primary_source);
		terma_primary += prim_term;

		// Scatter/extra-focal term: inverse-square from scatter source
		float scat_term = wS * atten * sqr(beam->model.scatter_src_dist / distance_to_scatter_source);
		terma_scatter += scat_term;
	}

	// Combine primary and scatter with respective head transmissions and scatter weight
	float terma_combined = (1.0f - beam->model.scatter_src_weight) * primary_transmission * terma_primary
						 + beam->model.scatter_src_weight * scatter_transmission * terma_scatter;

	// Electron contamination (empirical), keep dependence on primary transmission
	float transmission_ratio = fminf(1.0f, scatter_transmission / fmaxf(primary_transmission, 1e-6f));
	electron = fmaxf(0.0f, (expf(-beam->model.electron_attenuation * wet) - expf(-beam->model.electron_attenuation * beam->model.electron_fitted_dmax)) / (1.0f - expf(-beam->model.electron_attenuation * beam->model.electron_fitted_dmax)));

	TERMAArray[vox_index] = off_axis_factor * terma_combined;
	ElectronArray[vox_index] = beam->model.electron_src_weight * (0.4f + (0.3f * transmission_ratio)) * electron * primary_transmission;

}

__global__ void cccKernel(IMRTDose * dose, IMRTBeam * beam, Texture3D TERMATexture, Texture3D DensityTexture, float * ElectronArray){

	PointIJK vox_ijk;
	vox_ijk.k = threadIdx.x + (blockIdx.x * blockDim.x);
	vox_ijk.j = threadIdx.y + (blockIdx.y * blockDim.y);
	vox_ijk.i = threadIdx.z + (blockIdx.z * blockDim.z);

	if(!dose->pointIJKWithinImage(&vox_ijk)) {
		return;
	}

	unsigned vox_index = dose->pointIJKtoIndex(&vox_ijk);

	PointXYZ vox_img_xyz;
	dose->pointIJKtoXYZ(&vox_ijk, &vox_img_xyz, beam);

	PointXYZ tex_img_xyz;
	dose->pointXYZtoTextureXYZ(&vox_img_xyz, &tex_img_xyz, beam);

	if (TERMATexture.sample(tex_img_xyz) <= 0.01f){
		dose->DoseArray[vox_index] = 0.0f;
		return;
	}

	PointXYZ vox_head_xyz;
	beam->pointXYZImageToHead(&vox_img_xyz, &vox_head_xyz);

	float dose_value = 0.0f;
	const float step_cm = dose->spacing * 0.1f; // cm

	__shared__ struct {
		float cosx, sinx;
	} trig[12];

	for (int i = 0; i < 12; i++) {
		sincosf((float)i * CUDART_PI_F / 6.0f, &trig[i].sinx, &trig[i].cosx);
	}

	for(int i = 0; i < 6; i++){

		// Select depth-dependent kernel parameters if enabled
		const float wet_here = dose->WETArray[vox_index]; // g/cm^2
		const float *kernel_base = beam->model.kernel;

		if (beam->model.use_depth_dependent_kernel && beam->model.kernel_depths != nullptr
			&& beam->model.kernel_params != nullptr && beam->model.n_kernel_depths > 0) {
			int bin = lowerBound(beam->model.kernel_depths, beam->model.n_kernel_depths, wet_here);
			if (bin > 0) { bin -= 1; }
			bin = min(bin, beam->model.n_kernel_depths - 1);
			// Each bin holds 6 angles × 4 params
			kernel_base = beam->model.kernel_params + (bin * 24);
		}

		float th = beam->model.kernel[i] * CUDART_PI_F / 180.0f;
		// Depth-dependent parameters override when available
		float Am = kernel_base ? kernel_base[(beam->model.use_depth_dependent_kernel ? i * 4 + 0 : i + 6)] : 0.0f;
		float am = kernel_base ? kernel_base[(beam->model.use_depth_dependent_kernel ? i * 4 + 1 : i + 12)] : 0.0f;
		float Bm = kernel_base ? kernel_base[(beam->model.use_depth_dependent_kernel ? i * 4 + 2 : i + 18)] : 0.0f;
		float bm = kernel_base ? kernel_base[(beam->model.use_depth_dependent_kernel ? i * 4 + 3 : i + 24)] : 0.0f;
		float ray_length_init = beam->model.kernel[i + 30];
		float dir_weight;
		if (beam->model.kernel_weights != nullptr) {
			dir_weight = beam->model.kernel_weights[i] / 12.0f; // spread across azimuths
		} else {
			dir_weight = 1.0f / (6.0f * 12.0f); // uniform quadrature default
		}

		const auto sinth = sinf(th), costh = cosf(th);

		for(int j = 0; j < 12; j++){

			PointXYZ tangent_head_xyz;
			tangent_head_xyz.x = sinth * trig[j].cosx;
			tangent_head_xyz.y = sinth * trig[j].sinx;
			tangent_head_xyz.z = costh;

			PointXYZ tangent_img_xyz;
			beam->pointXYZHeadToImage(&tangent_head_xyz, &tangent_img_xyz);
			beam->kernelTilt(&vox_img_xyz, &tangent_img_xyz);

			float Rs = 0.0f, Rp = 0.0f, Ti = 0.0f;
			float Ti_hist = 0.0f;
			bool first_sample = true;

			for (float ray_length = ray_length_init; ray_length >= 0.0f; ray_length -= step_cm) {

				PointXYZ ray_img_xyz;
				ray_img_xyz.x = fmaf(tangent_img_xyz.x, ray_length * 10.0f, vox_img_xyz.x);
				ray_img_xyz.y = fmaf(tangent_img_xyz.y, ray_length * 10.0f, vox_img_xyz.y);
				ray_img_xyz.z = fmaf(tangent_img_xyz.z, ray_length * 10.0f, vox_img_xyz.z);

				dose->pointXYZtoTextureXYZ(&ray_img_xyz, &tex_img_xyz, beam);
				Ti = TERMATexture.sample(tex_img_xyz);
				float Di = fmaxf(DensityTexture.sample(tex_img_xyz), AIR_DENSITY) * step_cm;

				// Optional heterogeneity smoothing (history correction)
				float Ti_eff;
				if (beam->model.heterogeneity_alpha > 0.0f) {
					if (first_sample) {
						Ti_hist = Ti;
						first_sample = false;
					} else {
						Ti_hist = fmaf(beam->model.heterogeneity_alpha, Ti, (1.0f - beam->model.heterogeneity_alpha) * Ti_hist);
					}
					Ti_eff = Ti_hist;
				} else {
					Ti_eff = Ti;
				}

				// Primary: exponencial estável
				const auto expon_p = expf(-am * Di);
				Rp = Rp * expon_p + (Ti_eff * sinth * (Am / (am * am)) * (1.0f - expon_p));
				
				// Scatter: usar distância radiológica uma única vez
				const auto expon_s = expf(-bm * Di);
				Rs = Rs * expon_s + (Ti_eff * Di * sinth * (Bm / (bm * bm)) * (1.0f - expon_s));

			}

			dose_value += dir_weight * (am * Rp + bm * Rs);

		}
	}

	dose_value += ElectronArray[vox_index];

	if(!isnan(dose_value) && (dose_value >= 0.0f)){
		dose->DoseArray[vox_index] = beam->model.mu_cal * dose_value * beam->mu;
	}

}


void photon_dose_cuda(int gpu_id, IMRTDose * h_dose, IMRTBeam * h_beam){

	CUDA_CHECK(cudaSetDevice(gpu_id));

	IMRTDose d_dose(h_dose);
	IMRTBeam d_beam(h_beam);

	DevicePointer<float> DoseArray(MemoryTag::Zeroed(), h_dose->num_voxels);
	DevicePointer<float> WETArray(MemoryTag::Zeroed(), h_dose->num_voxels);
	DevicePointer<float> TERMAArray(MemoryTag::Zeroed(), h_dose->num_voxels);
	DevicePointer<float> ElectronArray(MemoryTag::Zeroed(), h_dose->num_voxels);

	d_dose.WETArray = WETArray.get();
	d_dose.DoseArray = DoseArray.get();

	DevicePointer<MLCPair> MLCPairArray(h_beam->mlc, h_beam->n_mlc_pairs);
	DevicePointer<float> d_profile_radius(h_beam->model.profile_radius, h_beam->model.n_profile_points);
	DevicePointer<float> d_profile_intensities(h_beam->model.profile_intensities, h_beam->model.n_profile_points);
	DevicePointer<float> d_profile_softening(h_beam->model.profile_softening, h_beam->model.n_profile_points);
	DevicePointer<float> d_spectrum_attenuation_coefficients(h_beam->model.spectrum_attenuation_coefficients, h_beam->model.n_spectral_energies);
	DevicePointer<float> d_spectrum_primary_weights(h_beam->model.spectrum_primary_weights, h_beam->model.n_spectral_energies);
	DevicePointer<float> d_spectrum_scatter_weights(h_beam->model.spectrum_scatter_weights, h_beam->model.n_spectral_energies);
	DevicePointer<float> d_kernel(h_beam->model.kernel, h_beam->model.kernel_len);
	std::unique_ptr<DevicePointer<float>> d_kernel_weights;
	std::unique_ptr<DevicePointer<float>> d_kernel_depths;
	std::unique_ptr<DevicePointer<float>> d_kernel_params;

	// Default nulls for optional kernel data
	d_beam.model.kernel_weights = nullptr;
	d_beam.model.kernel_depths = nullptr;
	d_beam.model.kernel_params = nullptr;

	if (h_beam->model.kernel_weights != nullptr) {
		d_kernel_weights = std::make_unique<DevicePointer<float>>(h_beam->model.kernel_weights, 6);
		d_beam.model.kernel_weights = d_kernel_weights->get();
	}
	if (h_beam->model.use_depth_dependent_kernel && h_beam->model.kernel_depths != nullptr && h_beam->model.kernel_params != nullptr) {
		d_kernel_depths = std::make_unique<DevicePointer<float>>(h_beam->model.kernel_depths, h_beam->model.n_kernel_depths);
		// kernel_params length = n_depths * 24 (6 angles * 4 params)
		d_kernel_params = std::make_unique<DevicePointer<float>>(h_beam->model.kernel_params, h_beam->model.n_kernel_depths * 24);
		d_beam.model.kernel_depths = d_kernel_depths->get();
		d_beam.model.kernel_params = d_kernel_params->get();
	}

	d_beam.mlc = MLCPairArray.get();
	d_beam.model.profile_radius = d_profile_radius.get();
	d_beam.model.profile_intensities = d_profile_intensities.get();
	d_beam.model.profile_softening = d_profile_softening.get();
	d_beam.model.spectrum_attenuation_coefficients = d_spectrum_attenuation_coefficients.get();
	d_beam.model.spectrum_primary_weights = d_spectrum_primary_weights.get();
	d_beam.model.spectrum_scatter_weights = d_spectrum_scatter_weights.get();
	d_beam.model.kernel = d_kernel.get();
	d_beam.model.kernel_len = h_beam->model.kernel_len;
	d_beam.model.use_depth_dependent_kernel = h_beam->model.use_depth_dependent_kernel;
	d_beam.model.n_kernel_depths = h_beam->model.n_kernel_depths;

	// CUDA_CHECK(cudaMemcpyToSymbol(g_kernel, h_kernel, 6 * 6 * sizeof(float)));

	DevicePointer<IMRTBeam> d_beam_ptr(&d_beam);
	DevicePointer<IMRTDose> d_dose_ptr(&d_dose);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((d_dose.img_sz.k + TILE_WIDTH - 1) / TILE_WIDTH, (d_dose.img_sz.j + TILE_WIDTH - 1) / TILE_WIDTH, (d_dose.img_sz.i + TILE_WIDTH - 1) / TILE_WIDTH);

	auto DensityTexture = Texture3D::fromHostData(h_dose->DensityArray, h_dose->img_sz, cudaFilterModeLinear, AIR_DENSITY);

    rayTraceKernel<<<dimGrid, dimBlock>>>(d_dose_ptr, d_beam_ptr, DensityTexture);
    termaKernel<<<dimGrid, dimBlock>>>(d_dose_ptr, d_beam_ptr, TERMAArray, ElectronArray);

	auto TERMATexture = Texture3D::fromDeviceData(TERMAArray, h_dose->img_sz, cudaFilterModeLinear);

	cccKernel<<<dimGrid, dimBlock>>>(d_dose_ptr, d_beam_ptr, TERMATexture, DensityTexture, ElectronArray);

	CUDA_CHECK(cudaMemcpy(h_dose->DoseArray, d_dose.DoseArray, d_dose.num_voxels * sizeof(float), cudaMemcpyDeviceToHost));

}

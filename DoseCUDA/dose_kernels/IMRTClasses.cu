#include "./IMRTClasses.cuh"
#include "MemoryClasses.h"


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
	
	// ========== ETAPA 1: JAWS (interseção retangular divergente) ==========
	float jaw_transmission_factor = 1.0f;
	
	if (model.has_xjaws || model.has_yjaws) {
		// Projetar jaws do plano jaw/MLC para plano do voxel (divergente)
		const float xLeft_jaw  = ((xjaws[0] * mlc_scale - point_xyz->x) * divergence_scale) + point_xyz->x;
		const float xRight_jaw = ((xjaws[1] * mlc_scale - point_xyz->x) * divergence_scale) + point_xyz->x;
		const float yBot_jaw   = ((yjaws[0] * mlc_scale - point_xyz->y) * divergence_scale) + point_xyz->y;
		const float yTop_jaw   = ((yjaws[1] * mlc_scale - point_xyz->y) * divergence_scale) + point_xyz->y;
		
		// Fração exposta pela jaw (gaussiana integrada)
		const float exposedSourceX_jaw = 0.5f * (erff(xRight_jaw * invSqrt2_x) - erff(xLeft_jaw * invSqrt2_x));
		const float exposedSourceY_jaw = 0.5f * (erff(yTop_jaw * invSqrt2_y) - erff(yBot_jaw * invSqrt2_y));
		
		float jaw_open_fraction = exposedSourceX_jaw * exposedSourceY_jaw;
		
		// Transmissão: se jaw fechada (fração < limiar), aplicar leakage
		jaw_transmission_factor = fmaxf(jaw_open_fraction, model.jaw_transmission);
	}
	
	// ========== ETAPA 2: MLC (folhas leaf-by-leaf) ==========
	float mlc_transmission = 0.0f;

    for (int i = 0; i < this->n_mlc_pairs; ++i) {

        const float yBottom = (this->mlc[i].y_offset - 0.5f * this->mlc[i].y_width) * mlc_scale;
        const float yTop    = (this->mlc[i].y_offset + 0.5f * this->mlc[i].y_width) * mlc_scale;

		const float xLeft  = (this->mlc[i].x1 * mlc_scale);
		const float xRight = (this->mlc[i].x2 * mlc_scale);

		const float tipMLC1 = ((xLeft - point_xyz->x) * divergence_scale) + point_xyz->x;
		const float tipMLC2 = ((xRight - point_xyz->x) * divergence_scale) + point_xyz->x;

		const float edgeMLC1 = ((yBottom - point_xyz->y) * divergence_scale) + point_xyz->y;
		const float edgeMLC2 = ((yTop - point_xyz->y) * divergence_scale) + point_xyz->y;

		const float exposedSourceX = 0.5f * (erff(tipMLC2 * invSqrt2_x) - erff(tipMLC1 * invSqrt2_x));
		const float exposedSourceY = 0.5f * (erff(edgeMLC2 * invSqrt2_y) - erff(edgeMLC1 * invSqrt2_y));

		float leaf_pair_open_fraction = exposedSourceX * exposedSourceY;
		
		// Aplicar leakage MLC: quando fechado, não é zero
		mlc_transmission += fmaxf(leaf_pair_open_fraction, model.mlc_transmission);
    }

    // ========== ETAPA 3: COMBINAÇÃO (jaws ∩ MLC) ==========
    // Transmissão final = jaws * MLC (geometria em série)
    return jaw_transmission_factor * mlc_transmission;

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

__device__ void IMRTBeam::interpolateKernelParams(int angle_idx, float z_prime, float * Am, float * am, float * Bm, float * bm){
	
	// FASE 2: Interpolar parâmetros do kernel baseado na profundidade z' (WET)
	
	// Se não usar kernel z-dependente, usar kernel fixo (fallback)
	if (!model.use_depth_dependent_kernel || model.kernel_depths == nullptr) {
		*Am = model.kernel[angle_idx + 6];
		*am = model.kernel[angle_idx + 12];
		*Bm = model.kernel[angle_idx + 18];
		*bm = model.kernel[angle_idx + 24];
		return;
	}
	
	// Clampar z_prime dentro dos limites da LUT
	z_prime = fmaxf(model.kernel_depths[0], fminf(z_prime, model.kernel_depths[model.n_kernel_depths - 1]));
	
	// Buscar índice inferior (binary search implícita via lowerBound)
	int depth_idx = lowerBound(model.kernel_depths, model.n_kernel_depths, z_prime);
	
	// Casos extremos: fora dos limites
	if (depth_idx == model.n_kernel_depths) {
		// Além do último ponto: usar último valor
		int base_idx = (model.n_kernel_depths - 1) * 24 + angle_idx * 4;
		*Am = model.kernel_params[base_idx + 0];
		*am = model.kernel_params[base_idx + 1];
		*Bm = model.kernel_params[base_idx + 2];
		*bm = model.kernel_params[base_idx + 3];
		return;
	}
	
	if (depth_idx == 0) {
		// Antes do primeiro ponto: usar primeiro valor
		int base_idx = angle_idx * 4;
		*Am = model.kernel_params[base_idx + 0];
		*am = model.kernel_params[base_idx + 1];
		*Bm = model.kernel_params[base_idx + 2];
		*bm = model.kernel_params[base_idx + 3];
		return;
	}
	
	// Interpolação linear entre depth_idx-1 e depth_idx
	float z0 = model.kernel_depths[depth_idx - 1];
	float z1 = model.kernel_depths[depth_idx];
	float t = (z_prime - z0) / (z1 - z0);  // Fator de interpolação [0, 1]
	
	// Índices na LUT
	int base_idx0 = (depth_idx - 1) * 24 + angle_idx * 4;
	int base_idx1 = depth_idx * 24 + angle_idx * 4;
	
	// Interpolar cada parâmetro
	*Am = fmaf(t, model.kernel_params[base_idx1 + 0] - model.kernel_params[base_idx0 + 0], model.kernel_params[base_idx0 + 0]);
	*am = fmaf(t, model.kernel_params[base_idx1 + 1] - model.kernel_params[base_idx0 + 1], model.kernel_params[base_idx0 + 1]);
	*Bm = fmaf(t, model.kernel_params[base_idx1 + 2] - model.kernel_params[base_idx0 + 2], model.kernel_params[base_idx0 + 2]);
	*bm = fmaf(t, model.kernel_params[base_idx1 + 3] - model.kernel_params[base_idx0 + 3], model.kernel_params[base_idx0 + 3]);
}


__host__ IMRTDose::IMRTDose(CudaDose * h_dose) : CudaDose(h_dose) {}


__global__ void termaKernel(IMRTDose * dose, IMRTBeam * beam, float * TERMAPrimaryArray, float * TERMAExtrafocalArray, float * ElectronArray){

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

	// ==============================================
	// FASE 1: SEPARAÇÃO CORRETA DO TERMA
	// ==============================================
	// Separar componentes: PRIMARY (bloqueado pelo MLC) vs EXTRAFOCAL (sempre transmitido)
	
	float terma_primary = 0.f;
	float terma_extrafocal = 0.f;
	float electron = 0.f;

	// Loop sobre espectro policromático
	for(int i = 0; i < beam->model.n_spectral_energies; i++){
		// PRIMARY: fonte principal, atenuado pelo paciente, bloqueado pelo MLC
		terma_primary += beam->model.spectrum_primary_weights[i] * 
			expf(-beam->model.spectrum_attenuation_coefficients[i] * wet * off_axis_softening) * 
			sqr(beam->model.primary_src_dist / distance_to_primary_source);
		
		// EXTRAFOCAL: fonte secundária (scatter head), sempre transmitido, não bloqueado pelo MLC
		terma_extrafocal += beam->model.spectrum_scatter_weights[i] * 
			expf(-beam->model.spectrum_attenuation_coefficients[i] * wet * off_axis_softening) * 
			sqr(beam->model.scatter_src_dist / distance_to_scatter_source);
	}

	// Elétrons contaminantes (superficiais)
	electron = fmaxf(0.0f, (expf(-beam->model.electron_attenuation * wet) - 
		expf(-beam->model.electron_attenuation * beam->model.electron_fitted_dmax)) / 
		(1.0f - expf(-beam->model.electron_attenuation * beam->model.electron_fitted_dmax)));

	// CRÍTICO: Aplicar transmissão SOMENTE ao primário
	// Primary: bloqueado pelo MLC (usa primary_transmission)
	TERMAPrimaryArray[vox_index] = off_axis_factor * primary_transmission * terma_primary;
	
	// Extrafocal: NÃO é bloqueado pelo MLC (peso global apenas)
	TERMAExtrafocalArray[vox_index] = off_axis_factor * beam->model.scatter_src_weight * terma_extrafocal;
	
	// Elétrons: modulados por transmissão e scatter_transmission
	float transmission_ratio = fminf(1.00f, scatter_transmission / fmaxf(primary_transmission, 1e-6f));
	ElectronArray[vox_index] = beam->model.electron_src_weight * 
		(0.4f + (0.3f * transmission_ratio)) * electron * primary_transmission;

}

__global__ void cccKernel(IMRTDose * dose, IMRTBeam * beam, Texture3D TERMAPrimaryTexture, Texture3D TERMAExtrafocalTexture, Texture3D DensityTexture, float * ElectronArray){

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

	// Checar se há TERMA suficiente (primary + extrafocal)
	float terma_check = TERMAPrimaryTexture.sample(tex_img_xyz) + TERMAExtrafocalTexture.sample(tex_img_xyz);
	if (terma_check <= 0.01f){
		dose->DoseArray[vox_index] = 0.0f;
		return;
	}

	PointXYZ vox_head_xyz;
	beam->pointXYZImageToHead(&vox_img_xyz, &vox_head_xyz);

	float dose_value = 0.0f;
	float sp = dose->spacing / 10.0f; //cm

	__shared__ struct {
		float cosx, sinx;
	} trig[12];

	for (int i = 0; i < 12; i++) {
		sincosf((float)i * CUDART_PI_F / 6.0f, &trig[i].sinx, &trig[i].cosx);
	}

	for(int i = 0; i < 6; i++){

		float th = beam->model.kernel[i] * CUDART_PI_F / 180.0f;
		
		// FASE 2: Calcular z' (profundidade em WET) do voxel atual
		float z_prime = dose->WETArray[vox_index] / 10.0f;  // Converter mm para cm
		
		// FASE 2: Interpolar parâmetros do kernel baseado em z'
		float Am, am, Bm, bm;
		beam->interpolateKernelParams(i, z_prime, &Am, &am, &Bm, &bm);
		
		float ray_length_init = beam->model.kernel[i + 30];

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
			float Di = AIR_DENSITY * sp;
			float ray_length = ray_length_init;

			while(ray_length >= 0.0f) {

				PointXYZ ray_img_xyz;
				ray_img_xyz.x = fmaf(tangent_img_xyz.x, ray_length * 10.0f, vox_img_xyz.x);
				ray_img_xyz.y = fmaf(tangent_img_xyz.y, ray_length * 10.0f, vox_img_xyz.y);
				ray_img_xyz.z = fmaf(tangent_img_xyz.z, ray_length * 10.0f, vox_img_xyz.z);

				dose->pointXYZtoTextureXYZ(&ray_img_xyz, &tex_img_xyz, beam);
				
				// FASE 1: Somar primary + extrafocal separadamente
				float Ti_primary = TERMAPrimaryTexture.sample(tex_img_xyz);
				float Ti_extrafocal = TERMAExtrafocalTexture.sample(tex_img_xyz);
				Ti = Ti_primary + Ti_extrafocal;
				
				Di = DensityTexture.sample(tex_img_xyz);

				Di = fmaxf(Di, AIR_DENSITY) * sp;
				
				// FASE 3: Lateral density scaling correto
				// Densidade relativa para scaling lateral (ρ/ρ_água)
			float rho_rel = Di / (1.0f * sp);  // Densidade relativa (água = 1.0 g/cc)
			
			// Primary component (exponencial estável)
			const auto expon_p = expf(-am * Di);
			Rp = Rp * expon_p + (Ti * sinth * (Am / (am * am)) * (1.0f - expon_p));
			
			// Scatter component (CORRIGIDO: exponencial estável, não linear)
			// AAA/Eclipse descrevem scatter também como exponencial, apenas com parâmetros diferentes
			const auto expon_s = expf(-bm * Di * rho_rel);
			Rs = Rs * expon_s + (Ti * Di * sinth * (Bm / (bm * bm)) * (1.0f - expon_s) * rho_rel);

			dose_value += am * Rp + bm * Rs;

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
	
	// FASE 1: Arrays separados para TERMA primary e extrafocal
	DevicePointer<float> TERMAPrimaryArray(MemoryTag::Zeroed(), h_dose->num_voxels);
	DevicePointer<float> TERMAExtrafocalArray(MemoryTag::Zeroed(), h_dose->num_voxels);
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
	DevicePointer<float> d_kernel(h_beam->model.kernel, 6 * 6);

	d_beam.mlc = MLCPairArray.get();
	d_beam.model.profile_radius = d_profile_radius.get();
	d_beam.model.profile_intensities = d_profile_intensities.get();
	d_beam.model.profile_softening = d_profile_softening.get();
	d_beam.model.spectrum_attenuation_coefficients = d_spectrum_attenuation_coefficients.get();
	d_beam.model.spectrum_primary_weights = d_spectrum_primary_weights.get();
	d_beam.model.spectrum_scatter_weights = d_spectrum_scatter_weights.get();
	d_beam.model.kernel = d_kernel.get();
	
	// FASE 2: Copiar kernel z-dependente para device (se disponível)
	if (h_beam->model.use_depth_dependent_kernel && h_beam->model.kernel_depths != nullptr) {
		DevicePointer<float> d_kernel_depths(h_beam->model.kernel_depths, h_beam->model.n_kernel_depths);
		DevicePointer<float> d_kernel_params(h_beam->model.kernel_params, h_beam->model.n_kernel_depths * 24);
		d_beam.model.kernel_depths = d_kernel_depths.get();
		d_beam.model.kernel_params = d_kernel_params.get();
	} else {
		d_beam.model.kernel_depths = nullptr;
		d_beam.model.kernel_params = nullptr;
	}

	DevicePointer<IMRTBeam> d_beam_ptr(&d_beam);
	DevicePointer<IMRTDose> d_dose_ptr(&d_dose);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((d_dose.img_sz.k + TILE_WIDTH - 1) / TILE_WIDTH, (d_dose.img_sz.j + TILE_WIDTH - 1) / TILE_WIDTH, (d_dose.img_sz.i + TILE_WIDTH - 1) / TILE_WIDTH);

	auto DensityTexture = Texture3D::fromHostData(h_dose->DensityArray, h_dose->img_sz, cudaFilterModeLinear, AIR_DENSITY);

    rayTraceKernel<<<dimGrid, dimBlock>>>(d_dose_ptr.get(), d_beam_ptr.get(), DensityTexture);
    
    // FASE 1: Chamar termaKernel com arrays separados
    termaKernel<<<dimGrid, dimBlock>>>(d_dose_ptr.get(), d_beam_ptr.get(), TERMAPrimaryArray.get(), TERMAExtrafocalArray.get(), ElectronArray.get());

	// FASE 1: Criar texturas separadas
	auto TERMAPrimaryTexture = Texture3D::fromDeviceData(TERMAPrimaryArray, h_dose->img_sz, cudaFilterModeLinear);
	auto TERMAExtrafocalTexture = Texture3D::fromDeviceData(TERMAExtrafocalArray, h_dose->img_sz, cudaFilterModeLinear);

	cccKernel<<<dimGrid, dimBlock>>>(d_dose_ptr.get(), d_beam_ptr.get(), TERMAPrimaryTexture, TERMAExtrafocalTexture, DensityTexture, ElectronArray.get());

	CUDA_CHECK(cudaMemcpy(h_dose->DoseArray, d_dose.DoseArray, d_dose.num_voxels * sizeof(float), cudaMemcpyDeviceToHost));

}

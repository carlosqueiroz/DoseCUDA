#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <object.h>
#include <numpy/arrayobject.h>
#include <cstdint>

#include "IMRTClasses.cuh"
#include "MemoryClasses.h"
#include "GammaKernels.cuh"


/** @brief Check a NumPy array for dimensionality and contained element type
 * 	@param arr
 * 		NumPy array pointer
 * 	@param dim
 * 		Required dimensionality
 * 	@param type
 * 		Required type constant
 * 	@returns true if the constraint is satisfied, false if not
 */
static bool pyarray_typecheck(const PyArrayObject *arr, int dim, int type) {

	return PyArray_NDIM(arr) == dim && PyArray_TYPE(arr) == type;
}


/** @brief Fetch the array's data pointer as a pointer to T */
template <class T>
static T *pyarray_as(PyArrayObject *arr) {

	return reinterpret_cast<T *>(PyArray_DATA(arr));
}


/** @brief Borrow a reference to a NumPy array object contained by a class
 * 	@param self
 * 		Class containing the array
 * 	@param attr
 * 		Member/field name of the array
 * 	@param dim
 * 		Dimensionality/rank of the array
 * 	@param[out] arr
 * 		The `PyArrayObject *` will be written here on success. You do not need
 * 		to `Py_DECREF` this object
 * 	@returns true on success, false on error. If an error occurs, a Python
 * 		exception will already have been raised
 */
static bool pyobject_getarray(PyObject *self, const char *attr, int dim, PyArrayObject **arr) {

	PyObject *ptr = PyObject_GetAttrString(self, attr);
	if (!ptr) {
		return false;
	}

	bool result = false;
	*arr = reinterpret_cast<PyArrayObject *>(ptr);
	if (PyArray_Check(ptr) && pyarray_typecheck(*arr, dim, NPY_FLOAT)) {
		result = true;
	} else {
		PyErr_Format(PyExc_ValueError, "'%s' must be %d-dimensional and of type float.", attr, dim);
	}
	Py_DECREF(ptr);
	return result;
}


/** @brief Get a `double` from a Python class
 * 	@param self
 * 		Class
 * 	@param attr
 * 		Member/field name of the float
 * 	@param[out] value
 * 		The result will be written here on success
 * 	@returns true on success, false on error. If an error occurs, a Python
 * 		exception will have been raised
 */
static bool pyobject_getfloat(PyObject *self, const char *attr, double *value) {

	PyObject *ptr = PyObject_GetAttrString(self, attr);
	if (!ptr) {
		return false;
	}
	*value = PyFloat_AsDouble(ptr);
	Py_DECREF(ptr);
	return *value == -1.0 ? !PyErr_Occurred() : true;
}


static bool pyobject_getbool(PyObject *self, const char *attr, bool *value) {

	PyObject *ptr = PyObject_GetAttrString(self, attr);
	if (!ptr) {
		return false;
	}
	int result = PyObject_IsTrue(ptr);
	Py_DECREF(ptr);
	*value = result > 0;
	return result >= 0;
}


static void make_mlc_array(PyArrayObject *mlc, HostPointer<MLCPair> &res)
{
	const size_t count = PyArray_DIM(mlc, 0);
	const float *src = pyarray_as<float>(mlc);

	for (size_t i = 0; i < count; i++) {
		res[i].x1 = src[i];
		res[i].x2 = src[i + count];
		res[i].y_offset = src[i + 2 * count];
		res[i].y_width = src[i + 3 * count];
		// printf("MLC Pair %d: x1: %f, x2: %f, y_offset: %f, y_width: %f\n", i, res[i].x1, res[i].x2, res[i].y_offset, res[i].y_width);
	}
}


static PyObject * photon_dose(PyObject* self, PyObject* args) {

	PyObject *model_instance, *volume_instance, *cp_instance;
	int gpu_id;	

	// parse arguments
    if (!PyArg_ParseTuple(args, "OOOi", &model_instance, &volume_instance, &cp_instance, &gpu_id))
        return NULL;

	// check beam model properties 
double mu_cal, 
	primary_source_distance, 
	scatter_source_distance, 
	primary_source_size, 
	scatter_source_size, 
	mlc_distance, 
	scatter_source_weight, 
	electron_attenuation, 
	electron_src_weight,
	electron_fitted_dmax,
	jaw_transmission,
	mlc_transmission,
	heterogeneity_alpha;

	if (!pyobject_getfloat(model_instance, "mu_calibration", &mu_cal)
	 || !pyobject_getfloat(model_instance, "primary_source_distance", &primary_source_distance)
	 || !pyobject_getfloat(model_instance, "scatter_source_distance", &scatter_source_distance)
	 || !pyobject_getfloat(model_instance, "primary_source_size", &primary_source_size)
	 || !pyobject_getfloat(model_instance, "scatter_source_size", &scatter_source_size)
	 || !pyobject_getfloat(model_instance, "mlc_distance", &mlc_distance)
 || !pyobject_getfloat(model_instance, "scatter_source_weight", &scatter_source_weight)
 || !pyobject_getfloat(model_instance, "electron_attenuation", &electron_attenuation)
 || !pyobject_getfloat(model_instance, "electron_source_weight", &electron_src_weight)
 || !pyobject_getfloat(model_instance, "electron_fitted_dmax", &electron_fitted_dmax)
 || !pyobject_getfloat(model_instance, "jaw_transmission", &jaw_transmission)
 || !pyobject_getfloat(model_instance, "mlc_transmission", &mlc_transmission)
 || !pyobject_getfloat(model_instance, "heterogeneity_alpha", &heterogeneity_alpha)) {
	return NULL;
}

	bool has_xjaws, 
		has_yjaws;
	if (!pyobject_getbool(model_instance, "has_xjaws", &has_xjaws)
	 || !pyobject_getbool(model_instance, "has_yjaws", &has_yjaws)) {
		return NULL;
	}

	PyArrayObject *profile_radius, 
		*profile_intensities, 
		*profile_softening, 
		*spectrum_attenuation_coefficients, 
		*spectrum_primary_weights, 
		*spectrum_scatter_weights, 
		*kernel,
		*kernel_weights = NULL,
		*kernel_depths = NULL,
		*kernel_params = NULL;
	if (!pyobject_getarray(model_instance, "profile_radius", 1, &profile_radius)
	 || !pyobject_getarray(model_instance, "profile_intensities", 1, &profile_intensities)
	 || !pyobject_getarray(model_instance, "profile_softening", 1, &profile_softening)
	 || !pyobject_getarray(model_instance, "spectrum_attenuation_coefficients", 1, &spectrum_attenuation_coefficients)
	 || !pyobject_getarray(model_instance, "spectrum_primary_weights", 1, &spectrum_primary_weights)
	 || !pyobject_getarray(model_instance, "spectrum_scatter_weights", 1, &spectrum_scatter_weights)
	 || !pyobject_getarray(model_instance, "kernel", 1, &kernel)) {
		return NULL;
	}
	// Optional kernel weights (length 6)
	PyObject *tmp_attr = PyObject_GetAttrString(model_instance, "kernel_weights");
	if (tmp_attr && PyArray_Check(tmp_attr) && pyarray_typecheck((PyArrayObject*)tmp_attr, 1, NPY_FLOAT)) {
		kernel_weights = (PyArrayObject*)tmp_attr;
	} else {
		Py_XDECREF(tmp_attr);
	}
	tmp_attr = PyObject_GetAttrString(model_instance, "kernel_depths");
	if (tmp_attr && PyArray_Check(tmp_attr) && pyarray_typecheck((PyArrayObject*)tmp_attr, 1, NPY_FLOAT)) {
		kernel_depths = (PyArrayObject*)tmp_attr;
	} else {
		Py_XDECREF(tmp_attr);
	}
	tmp_attr = PyObject_GetAttrString(model_instance, "kernel_params");
	if (tmp_attr && PyArray_Check(tmp_attr) && pyarray_typecheck((PyArrayObject*)tmp_attr, 1, NPY_FLOAT)) {
		kernel_params = (PyArrayObject*)tmp_attr;
	} else {
		Py_XDECREF(tmp_attr);
	}
	bool use_depth_dependent_kernel = false;
	pyobject_getbool(model_instance, "use_depth_dependent_kernel", &use_depth_dependent_kernel);

	// check volume data properties
	PyArrayObject *density_array, 
		*spacing_array, 
		*origin_array;
	if (!pyobject_getarray(volume_instance, "voxel_data", 3, &density_array)
	 || !pyobject_getarray(volume_instance, "spacing", 1, &spacing_array)
	 || !pyobject_getarray(volume_instance, "origin", 1, &origin_array)) {
		return NULL;
	}

	// check control point data properties
	PyArrayObject *iso_array, 
		*mlc_array,
		*xjaws_array,
		*yjaws_array;
	if (!pyobject_getarray(cp_instance, "iso", 1, &iso_array)
	 || !pyobject_getarray(cp_instance, "mlc", 2, &mlc_array)
	 || !pyobject_getarray(cp_instance, "xjaws", 1, &xjaws_array)
	 || !pyobject_getarray(cp_instance, "yjaws", 1, &yjaws_array)) {
		return NULL;
	}

	double mu, 
		ga, 
		ca, 
		ta;
	if (!pyobject_getfloat(cp_instance, "mu", &mu)
	 || !pyobject_getfloat(cp_instance, "ga", &ga)
	 || !pyobject_getfloat(cp_instance, "ca", &ca)
	 || !pyobject_getfloat(cp_instance, "ta", &ta)) {
		return NULL;
	}

	float * spacing = pyarray_as<float>(spacing_array);
	float * origin = pyarray_as<float>(origin_array);
	float * iso = pyarray_as<float>(iso_array);
	double voxel_sp = (double)spacing[0];

	size_t n_mlc_pairs = PyArray_DIM(mlc_array, 0);

	try {

		float adjusted_ga = fmodf(ga + 180.0f, 360.0f);

		size_t dims[3] = {
			(size_t)PyArray_DIMS(density_array)[0],
			(size_t)PyArray_DIMS(density_array)[1],
			(size_t)PyArray_DIMS(density_array)[2],
		};

		float adjusted_isocenter[3] = {
			iso[0] - origin[0],
			iso[1] - origin[1],
			iso[2] - origin[2]
		};

		// beam model object
		auto model = IMRTBeam::Model();
		model.n_profile_points = PyArray_DIM(profile_radius, 0);
		model.profile_radius = pyarray_as<float>(profile_radius);
		model.profile_intensities = pyarray_as<float>(profile_intensities);
		model.profile_softening = pyarray_as<float>(profile_softening);
		model.n_spectral_energies = PyArray_DIM(spectrum_attenuation_coefficients, 0);
		model.spectrum_attenuation_coefficients = pyarray_as<float>(spectrum_attenuation_coefficients);
		model.spectrum_primary_weights = pyarray_as<float>(spectrum_primary_weights);
		model.spectrum_scatter_weights = pyarray_as<float>(spectrum_scatter_weights);
		model.mu_cal = mu_cal;
		model.primary_src_dist = primary_source_distance;
		model.scatter_src_dist = scatter_source_distance;
		model.primary_src_size = primary_source_size;
		model.scatter_src_size = scatter_source_size;
		model.mlc_distance = mlc_distance;
		model.scatter_src_weight = scatter_source_weight;
	model.electron_attenuation = electron_attenuation;
	model.electron_src_weight = electron_src_weight;
	model.kernel = pyarray_as<float>(kernel);
	model.kernel_len = (int)PyArray_SIZE(kernel);
	model.kernel_weights = nullptr;
		if (kernel_weights) {
			model.kernel_weights = pyarray_as<float>(kernel_weights);
		}
		model.use_depth_dependent_kernel = use_depth_dependent_kernel;
		model.n_kernel_depths = 0;
		model.kernel_depths = nullptr;
		model.kernel_params = nullptr;
		if (use_depth_dependent_kernel && kernel_depths && kernel_params) {
			model.n_kernel_depths = (int)PyArray_DIM(kernel_depths, 0);
			model.kernel_depths = pyarray_as<float>(kernel_depths);
			model.kernel_params = pyarray_as<float>(kernel_params);
		}
		model.has_xjaws = has_xjaws;
		model.has_yjaws = has_yjaws;

		// If the control point explicitly contains jaw arrays, trust them and
		// enable the corresponding flags so headTransmission will apply jaws.
		if (xjaws_array != NULL) {
			if (PyArray_DIM(xjaws_array, 0) >= 2) {
				model.has_xjaws = true;
			}
		}
		if (yjaws_array != NULL) {
			if (PyArray_DIM(yjaws_array, 0) >= 2) {
				model.has_yjaws = true;
			}
		}
	model.electron_fitted_dmax = electron_fitted_dmax;
	model.jaw_transmission = jaw_transmission;
	model.mlc_transmission = mlc_transmission;
	model.heterogeneity_alpha = heterogeneity_alpha;

		// dose object
		IMRTDose dose_obj = IMRTDose(dims, voxel_sp);
		HostPointer<float> WETArray(dose_obj.num_voxels);
		HostPointer<float> DoseArray(dose_obj.num_voxels);

		dose_obj.DensityArray = pyarray_as<float>(density_array);
		dose_obj.WETArray = WETArray.get();
		dose_obj.DoseArray = DoseArray.get();

		// MLC object_array
		HostPointer<MLCPair> MLCPairArray(n_mlc_pairs);
		make_mlc_array(mlc_array, MLCPairArray);

		// beam object
		IMRTBeam beam_obj = IMRTBeam(adjusted_isocenter, adjusted_ga, ta, ca, &model);
		beam_obj.n_mlc_pairs = n_mlc_pairs;
		beam_obj.mlc = MLCPairArray.get();
		beam_obj.mu = mu;
		
		// Pass jaws from Python to CUDA
		float * xjaws = pyarray_as<float>(xjaws_array);
		float * yjaws = pyarray_as<float>(yjaws_array);
		beam_obj.xjaws[0] = xjaws[0];
		beam_obj.xjaws[1] = xjaws[1];
		beam_obj.yjaws[0] = yjaws[0];
		beam_obj.yjaws[1] = yjaws[1];

		// compute dose
    	photon_dose_cuda(gpu_id, &dose_obj, &beam_obj);

		PyObject *return_dose = PyArray_SimpleNewFromData(3, PyArray_DIMS(density_array), PyArray_TYPE(density_array), DoseArray.release());

		PyArray_ENABLEFLAGS((PyArrayObject*) return_dose, NPY_ARRAY_OWNDATA);

		if (kernel_weights) Py_DECREF(kernel_weights);
		if (kernel_depths) Py_DECREF(kernel_depths);
		if (kernel_params) Py_DECREF(kernel_params);

		return return_dose;

	} catch (std::bad_alloc &) {

		PyErr_SetString(PyExc_MemoryError, "Not enough host memory");

	} catch (std::runtime_error &e) {

		PyErr_Format(PyExc_RuntimeError, "CUDA error: %s", e.what());

	}

	return NULL;

}


/**
 * @brief Python wrapper for CUDA gamma computation
 * 
 * Arguments:
 *   dose_eval: 3D numpy array (float32) - evaluated dose
 *   dose_ref: 3D numpy array (float32) - reference dose
 *   spacing: tuple (sx, sy, sz) in mm
 *   dta_mm: float - distance-to-agreement in mm
 *   dd_percent: float - dose difference percentage
 *   dose_threshold_percent: float - threshold as % of max dose
 *   global_dose: float - global normalization dose (or 0 to use max of dose_ref)
 *   local: bool - use local normalization
 *   max_gamma: float - cap gamma values
 *   return_map: bool - whether to return gamma map
 *   gpu_id: int - GPU device ID
 * 
 * Returns:
 *   Dictionary with:
 *     - pass_rate: float
 *     - mean_gamma: float
 *     - gamma_p95: float
 *     - n_evaluated: int
 *     - n_passed: int
 *     - gamma_map: numpy array (if return_map=True)
 */
static PyObject* gamma_3d(PyObject* self, PyObject* args, PyObject* kwargs) {
	
	PyArrayObject *dose_eval_arr, *dose_ref_arr;
	PyObject *roi_obj = Py_None;
	PyObject *spacing_tuple;
	double dta_mm, dd_percent, dose_threshold_percent, global_dose, max_gamma, sampling = 1.0;
	int local_norm, return_map, gpu_id;
	
	static char* kwlist[] = {
		"dose_eval", "dose_ref", "spacing", 
		"dta_mm", "dd_percent", "dose_threshold_percent",
		"global_dose", "local", "max_gamma", "return_map", "gpu_id",
		"roi_mask", "sampling", NULL
	};
	
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!Oddddpdpi|Od", kwlist,
			&PyArray_Type, &dose_eval_arr,
			&PyArray_Type, &dose_ref_arr,
			&spacing_tuple,
			&dta_mm, &dd_percent, &dose_threshold_percent,
			&global_dose, &local_norm, &max_gamma, &return_map, &gpu_id,
			&roi_obj, &sampling)) {
		return NULL;
	}
	
	// Validate arrays
	if (!pyarray_typecheck(dose_eval_arr, 3, NPY_FLOAT)) {
		PyErr_SetString(PyExc_ValueError, "dose_eval must be 3D float32 array");
		return NULL;
	}
	if (!pyarray_typecheck(dose_ref_arr, 3, NPY_FLOAT)) {
		PyErr_SetString(PyExc_ValueError, "dose_ref must be 3D float32 array");
		return NULL;
	}
	
	// Check shapes match
	npy_intp *shape_eval = PyArray_DIMS(dose_eval_arr);
	npy_intp *shape_ref = PyArray_DIMS(dose_ref_arr);
	if (shape_eval[0] != shape_ref[0] || 
		shape_eval[1] != shape_ref[1] || 
		shape_eval[2] != shape_ref[2]) {
		PyErr_SetString(PyExc_ValueError, "dose_eval and dose_ref must have same shape");
		return NULL;
	}

	// ROI mask (optional)
	uint8_t* roi_data = nullptr;
	PyArrayObject* roi_mask_arr = nullptr;
	if (roi_obj != Py_None) {
		if (!PyArray_Check(roi_obj)) {
			PyErr_SetString(PyExc_ValueError, "roi_mask must be a NumPy array or None");
			return NULL;
		}
		roi_mask_arr = (PyArrayObject*)roi_obj;
		if (!pyarray_typecheck(roi_mask_arr, 3, NPY_BOOL) &&
			!pyarray_typecheck(roi_mask_arr, 3, NPY_UBYTE)) {
			PyErr_SetString(PyExc_ValueError, "roi_mask must be 3D bool or uint8 array");
			return NULL;
		}
		npy_intp *shape_mask = PyArray_DIMS(roi_mask_arr);
		if (shape_mask[0] != shape_ref[0] || shape_mask[1] != shape_ref[1] || shape_mask[2] != shape_ref[2]) {
			PyErr_SetString(PyExc_ValueError, "roi_mask shape must match dose arrays");
			return NULL;
		}
		roi_data = (uint8_t*)PyArray_DATA(roi_mask_arr);
	}
	
	// Parse spacing tuple
	if (!PyTuple_Check(spacing_tuple) || PyTuple_Size(spacing_tuple) != 3) {
		PyErr_SetString(PyExc_ValueError, "spacing must be tuple of 3 floats");
		return NULL;
	}
	double sx = PyFloat_AsDouble(PyTuple_GetItem(spacing_tuple, 0));
	double sy = PyFloat_AsDouble(PyTuple_GetItem(spacing_tuple, 1));
	double sz = PyFloat_AsDouble(PyTuple_GetItem(spacing_tuple, 2));
	
	// Get array data
	float* dose_eval_data = pyarray_as<float>(dose_eval_arr);
	float* dose_ref_data = pyarray_as<float>(dose_ref_arr);
	
	// Dimensions (z, y, x for numpy but we store as nx, ny, nz)
	int nz = (int)shape_ref[0];
	int ny = (int)shape_ref[1];
	int nx = (int)shape_ref[2];
	
	// Compute global dose if not specified
	if (global_dose <= 0) {
		size_t n_voxels = (size_t)nx * ny * nz;
		global_dose = 0;
		for (size_t i = 0; i < n_voxels; i++) {
			if (dose_ref_data[i] > global_dose) global_dose = dose_ref_data[i];
		}
	}
	
	// Prepare parameters
	GammaParams params;
	params.nx = nx;
	params.ny = ny;
	params.nz = nz;
	params.sx = (float)sx;
	params.sy = (float)sy;
	params.sz = (float)sz;
	params.dta_mm = (float)dta_mm;
	params.dd_percent = (float)dd_percent;
	params.dose_threshold_percent = (float)dose_threshold_percent;
	params.global_dose = (float)global_dose;
	params.local_normalization = (bool)local_norm;
	params.max_gamma = (float)max_gamma;
	params.sampling = (float)sampling;
	
	// Allocate gamma map if needed
	float* gamma_map_data = nullptr;
	PyObject* gamma_map_arr = nullptr;
	if (return_map) {
		gamma_map_arr = PyArray_SimpleNew(3, shape_ref, NPY_FLOAT);
		if (!gamma_map_arr) return NULL;
		gamma_map_data = (float*)PyArray_DATA((PyArrayObject*)gamma_map_arr);
	}
	
	// Prepare stats
	GammaStats stats;
	memset(&stats, 0, sizeof(GammaStats));
	
	try {
		// Set GPU
		cudaError_t err = cudaSetDevice(gpu_id);
		if (err != cudaSuccess) {
			throw std::runtime_error(cudaGetErrorString(err));
		}
		
		// Run gamma computation
		gamma_3d_cuda(dose_eval_data, dose_ref_data, roi_data, gamma_map_data, params, &stats, 0);
		
	} catch (std::runtime_error &e) {
		if (gamma_map_arr) Py_DECREF(gamma_map_arr);
		PyErr_Format(PyExc_RuntimeError, "CUDA gamma error: %s", e.what());
		return NULL;
	}
	
	// Calculate pass rate and mean
	double pass_rate = (stats.n_evaluated > 0) 
		? (double)stats.n_passed / stats.n_evaluated 
		: 0.0;
	double mean_gamma = (stats.n_evaluated > 0) 
		? stats.sum_gamma / stats.n_evaluated 
		: 0.0;
	
	// Calculate P95 from histogram
	double gamma_p95 = max_gamma;
	if (stats.n_evaluated > 0) {
		unsigned int target_count = (unsigned int)(0.95 * stats.n_evaluated);
		unsigned int cumsum = 0;
		for (int bin = 0; bin <= 100; bin++) {
			cumsum += stats.histogram[bin];
			if (cumsum >= target_count) {
				gamma_p95 = (bin / 100.0) * max_gamma;
				break;
			}
		}
	}
	
	// Build result dictionary
	PyObject* result = PyDict_New();
	PyDict_SetItemString(result, "pass_rate", PyFloat_FromDouble(pass_rate));
	PyDict_SetItemString(result, "mean_gamma", PyFloat_FromDouble(mean_gamma));
	PyDict_SetItemString(result, "gamma_p95", PyFloat_FromDouble(gamma_p95));
	PyDict_SetItemString(result, "n_evaluated", PyLong_FromUnsignedLong(stats.n_evaluated));
	PyDict_SetItemString(result, "n_passed", PyLong_FromUnsignedLong(stats.n_passed));
	
	if (gamma_map_arr) {
		PyDict_SetItemString(result, "gamma_map", gamma_map_arr);
		Py_DECREF(gamma_map_arr);  // Dict now owns reference
	}
	
	return result;
}


/**
 * @brief Check if CUDA gamma is available
 */
static PyObject* gamma_cuda_is_available(PyObject* self, PyObject* args) {
	if (gamma_cuda_available()) {
		Py_RETURN_TRUE;
	}
	Py_RETURN_FALSE;
}


static PyMethodDef DoseMethods[] = {
	{
		"photon_dose_cuda",
		photon_dose,
		METH_VARARGS,
		"Compute photon dose using collapsed cone convolution on the GPU."
	},
	{
		"gamma_3d_cuda",
		(PyCFunction)gamma_3d,
		METH_VARARGS | METH_KEYWORDS,
		"Compute 3D gamma index on GPU.\n\n"
		"Args:\n"
		"    dose_eval: 3D float32 array - evaluated dose\n"
		"    dose_ref: 3D float32 array - reference dose\n"
		"    spacing: tuple (sx, sy, sz) in mm\n"
		"    dta_mm: distance-to-agreement in mm\n"
		"    dd_percent: dose difference percentage (e.g., 3.0 for 3%)\n"
		"    dose_threshold_percent: threshold as % of global dose\n"
		"    global_dose: reference dose for normalization (0 = use max)\n"
		"    local: use local normalization (bool)\n"
		"    max_gamma: cap gamma values\n"
		"    roi_mask: optional bool mask array (same shape) to limit evaluation\n"
		"    sampling: sub-voxel sampling factor (1.0 = voxel)\n"
		"    return_map: return gamma map array (bool)\n"
		"    gpu_id: GPU device ID\n\n"
		"Returns:\n"
		"    dict with pass_rate, mean_gamma, gamma_p95, n_evaluated, n_passed, gamma_map"
	},
	{
		"gamma_cuda_available",
		gamma_cuda_is_available,
		METH_NOARGS,
		"Check if CUDA gamma computation is available."
	},
	{ 0 }
};


static struct PyModuleDef dosemodule = {
	PyModuleDef_HEAD_INIT,
	"dose_kernels",
	"Compute dose on the GPU.",
	-1,
	DoseMethods,
};


PyMODINIT_FUNC PyInit_dose_kernels(void) {
	import_array();
	return PyModule_Create(&dosemodule);
}

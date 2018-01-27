#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "nifti.h"
#include "fiberTrack.h"
#include <cstring>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <CL\opencl.h>
#include <ctime> 
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <time.h>
#include <stdlib.h>
using namespace std;

typedef struct coordinate_index
{ 
	int x;
	int y;
	int z;
} coor_idx;

bool same_voxel (coor_idx v1, coor_idx v2)
{
	return(v1.x == v2.x && v1.y == v2.y && v1.z == v2.z );
}

/* Return true if two voxels are not adjacent */
bool longdist_coor (coor_idx v1, coor_idx v2)
{
	return ( fabs((float)v1.x-v2.x) > 1 || fabs((float)v1.y-v2.y) > 1 || fabs((float)v1.z-v2.z) > 1 );
}

/*  Return the sequential index of a 3D coordinate */
int coor2index (coor_idx v, int dimX, int dimY)
{
	return(v.z*dimY*dimX + v.y*dimY + v.x);
}

inline int cub2ind(int x, int y, int z, int DimX, int DimY, int DimZ)
{
	return x + y * DimX + z * DimX * DimY;
}

template <typename T>
void reverse_y(T * src, int numSample, int dimX, int dimY, int dimZ)
{
	int size_cube = dimX * dimY * dimZ;
	for(int s = 0; s < numSample; s++)
	{
		for (int x = 0; x < dimX; x++)
		{
			for (int z = 0; z < dimZ; z++)
			{
				for (int y = 0; y < dimY/2; y++)
				{
					T temp = src[s * size_cube + cub2ind(x, y, z, dimX, dimY, dimZ)];
					src[s*size_cube + cub2ind(x, y, z, dimX, dimY, dimZ)] = src[s*size_cube + cub2ind( x, dimY-1-y, z, dimX, dimY, dimZ)];
					src[s*size_cube + cub2ind(x, dimY-1-y, z, dimX, dimY, dimZ)] = temp;
				}
			}
		}
	}
}




template <typename T>
void reverse_x(T * src, int numSample, int dimX, int dimY, int dimZ)
{
	int size_cube = dimX * dimY * dimZ;
	for(int s = 0; s < numSample; s++)
	{
		for (int y = 0; y < dimY; y++)
		{
			for (int z = 0; z < dimZ; z++)
			{
				for (int x = 0; x < dimX/2; x++)
				{
					T temp = src[s*size_cube + cub2ind(x, y, z, dimX, dimY, dimZ)];
					src[s*size_cube + cub2ind(x, y, z, dimX, dimY, dimZ)] = src[s*size_cube + cub2ind( dimX-1- x, y, z, dimX, dimY, dimZ)];
					src[s*size_cube + cub2ind(dimX-1-x, y, z, dimX, dimY, dimZ)] = temp;
				}
			}
		}
	}
}

//generate random x from a unit distribution of (0,N-1)
int randn(int n)  
{  
    int max = RAND_MAX - RAND_MAX % n;  
    int x;  
    do  
    {  
        x = rand();  
    }while ( x >= max );  
    return x % n;  
}  

// Host buffers for demo
// ********************************************************************* 

// OpenCL Vars
cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command que
cl_platform_id cpPlatform;      // OpenCL platform
cl_device_id cdDevice;          // OpenCL device
cl_program cpProgram;           // OpenCL program
cl_kernel ckKernel;             // OpenCL kernel
cl_event ceEvent;

cl_mem kernel_result;
cl_mem kernel_start_coordinate;
cl_mem kernel_start_direction;
//cl_mem kernel_step;
cl_mem kernel_iteration;
//cl_mem kernel_length;
cl_mem kernel_reason;
cl_mem f1_image;
cl_mem ph1_image;
cl_mem th1_image;
cl_mem f2_image;
cl_mem ph2_image;
cl_mem th2_image;
cl_image_format image_format;

size_t szGlobalWorkSize;        // 1D var for Total # of work items
size_t szWorkGroups;			// 1D var for Total # of work groups
size_t szLocalWorkSize;		    // 1D var for # of work items in the work group	
size_t szParmDataBytes;			// Byte size of context information
size_t szKernelLength;			// Byte size of kernel code
cl_int ciErr1, ciErr2;			// Error code var
char* cPathAndName = NULL;      // var for full paths to data, src, etc.
char* cSourceCL = NULL;         // Buffer to hold source for compilation 

// demo config vars
const int HdrLen = 352;

// Forward Declarations
// *********************************************************************
void Cleanup (int iExitCode);
string convertToString(const char *filename);

// Main function 
// *********************************************************************
int main(int argc, char **argv)
{
	// Varibles
	// ==============================================================================
	// input directory including bedpostx results
	char * input_dir = "D:/Programming/FiberTrackGPUtest/MinhuiData/0001S1.bedpostx";
	
	// make the macro definition of the USE_PROBMASK if a probability mask is used,
	// the probability mask can be derived from results of FSL ProbtrackX or from this projects
#ifdef USE_PROBMASK
	string probmask_file   = "probmap1.nii";
#endif

	#ifdef FINAL_SEED_TRACK
	string ROI_file  = "seed_cc.nii";
	#endif
	string string_input_filename_f1		= string(input_dir).append("/merged_f1samples.nii");
	string string_input_filename_f2		= string(input_dir).append("/merged_f2samples.nii");
	string string_input_filename_th1	= string(input_dir).append("/merged_th1samples.nii");
	string string_input_filename_th2	= string(input_dir).append("/merged_th2samples.nii");
	string string_input_filename_ph1	= string(input_dir).append("/merged_ph1samples.nii");
	string string_input_filename_ph2	= string(input_dir).append("/merged_ph2samples.nii");
	string string_input_filename_mask	= string(input_dir).append("/nodif_brain_mask.nii");
	
	#ifdef FINAL_SEED_TRACK
	string string_input_filename_seed	= string(input_dir).append("/").append(ROI_file);
	#endif
	
	#ifdef USE_PROBMASK
	string string_input_filename_probmask = string(input_dir).append("/").append(probmask_file)
	#endif
	
	ifstream filestream_input;

	// input data
	
#ifdef USE_PROBMASK
	int * array_input_probmask;
#endif


#ifdef FINAL_SEED_TRACK
	unsigned char *array_input_seed;
#endif
	short * array_input_mask;	// dimX * dimY * dimZ
	float * array_input_f1;		// dimX * dimY * dimZ * numSample
	float * array_input_th1;		//  dimX * dimY * dimZ * numSample
	float * array_input_ph1;		//  dimX * dimY * dimZ * numSample
	float * array_input_f2;		// dimX * dimY * dimZ * numSample
	float * array_input_th2;		//  dimX * dimY * dimZ * numSample
	float * array_input_ph2;		//  dimX * dimY * dimZ * numSample

	// dimensions
	int dimX, dimY, dimZ;
	float pixdimX, pixdimY, pixdimZ;
	int size_cube,size_cube_seed; 
	int numSample;
	int numVoxel; // number of mask == 1 
	int numValidVoxel; 

	// parameters
	
#ifdef USE_F_THRESHOLD_AS_TERMINATE_CONDITION
	float para_threshold_f = 0.2f;
#endif
	//float para_threshold_angular = 0.8f;    //move to .h file
	//int para_threshold_count = 150;     
	//int min_step = 50;					  //move to .h file

	// output files
	//char string_output_filename_sequence_merge[] = "prob_track";
	//char string_output_filename_nii[] = "prob_track.nii";
	//char string_output_filename_merge_nii[] = "prob_track_merge.nii";
	//char string_output_filename_info[] = "prob_track_info";
	ofstream filestream_output;

	// intermediate memory
	float * result_sequence; // numSeedVoxel * para_threshold_count
	int * step; // numSeedVoxel
	int * iteration; 
	int * length; // numSeedVoxel
	int * reason; // numSeedVoxel

	// mapping array
	//int * idx2cub;
	//int * cub2idx;
	float * start_coordinate;
	float * copy_start_coordinate;
	float* start_direction;

	//int * idx_valid2idx;

	clock_t CPU_preprocess_time, 
		GPU_preprocess_time, 
		process_time, 
		GPU_kernel_time, 
		GPU_transfer_time, 
		postprocess_time;
	clock_t total_GPU_kernel_time = 0, 
		total_GPU_transfer_time = 0, 
		total_postprocess_time = 0;
	//////////////////////////////////////////////////////////////
	//                       Timer                              //
	CPU_preprocess_time = clock(); 
	//                                                          //
	//////////////////////////////////////////////////////////////

	nii_hdr hdr;
	memset(&hdr, 0, sizeof (hdr)); 

	// read input files
	// ==============================================================================
	
	// read mask file...
	filestream_input.open(string_input_filename_mask.c_str(), ios::binary);     //nodif_brain_mask
	if (!filestream_input.good()) {cout<<"Can't open\t"<<string_input_filename_mask.c_str()<<endl;	system("pause");	return 1;}
	filestream_input.read((char*) &hdr, HdrLen);
	dimX = hdr.dim[1];
	dimY = hdr.dim[2];
	dimZ = hdr.dim[3];
	pixdimX = hdr.pixdim[1];
	pixdimY = hdr.pixdim[2];
	pixdimZ = hdr.pixdim[3];
	size_cube = dimX * dimY * dimZ; //size_cube is the number of all voxels
	array_input_mask = new short[size_cube];
	if (array_input_mask == NULL) {cout<<"Allocating memory failure."<<endl;	system("pause");	return 1;}
	filestream_input.read((char*)array_input_mask, sizeof(short) * size_cube);
	filestream_input.close();
	
	
	// read ROI file and check if the dimensions are accordant with mask.
	#ifdef FINAL_SEED_TRACK
	filestream_input.open(string_input_filename_seed.c_str(), ios::binary);     //nodif_brain_seed
	if (!filestream_input.good()) {cout<<"Can't open\t"<<string_input_filename_seed.c_str()<<endl;	system("pause");	return 1;}
	filestream_input.read((char*) (&hdr), HdrLen);
	if (dimX != hdr.dim[1] || dimY != hdr.dim[2] || dimZ != hdr.dim[3]) {cout<<"dimensions do not agree."<<endl; return 1;}	
	if (pixdimX != hdr.pixdim[1] || pixdimY != hdr.pixdim[2] || pixdimZ != hdr.pixdim[3]) {cout<<"pixdim do not agree."<<endl; return 1;}
	array_input_seed = new unsigned char[size_cube];
	if (array_input_seed == NULL) {cout<<"Allocating memory failure."<<endl;	system("pause");	return 1;}
	filestream_input.read((char*)array_input_seed, sizeof(unsigned char) * size_cube);
	filestream_input.close();
	#endif
	
	// read probmask file...	
	#ifdef USE_PROBMASK
	filestream_input.open(string_input_filename_probmask.c_str(), ios::binary);     //probmask
	if (!filestream_input.good()) {cout<<"Can't open\t"<<string_input_filename_probmask.c_str()<<endl;	system("pause");	return 1;}
	filestream_input.read((char*) &hdr, HdrLen);
	if (dimX != hdr.dim[1] || dimY != hdr.dim[2] || dimZ != hdr.dim[3]) {cout<<"dimensions do not agree."<<endl; return 1;}
	if (pixdimX != hdr.pixdim[1] || pixdimY != hdr.pixdim[2] || pixdimZ != hdr.pixdim[3]) {cout<<"pixdim do not agree."<<endl; return 1;}
	array_input_probmask = new int[size_cube];
	if (array_input_probmask == NULL) {cout<<"Allocating memory failure."<<endl;	system("pause");	return 1;}
	filestream_input.read((char*)array_input_probmask, sizeof(int) * size_cube);
	filestream_input.close();
	#endif
			
	// read merged f th ph sample files...
	filestream_input.open(string_input_filename_f1.c_str(), ios::binary);		//f1
	if (!filestream_input.good()) {cout<<"Can't open\t"<<string_input_filename_f1.c_str()<<endl;	system("pause");	return 1;}
	filestream_input.read((char*) &hdr, HdrLen);
	if (dimX != hdr.dim[1] || dimY != hdr.dim[2] || dimZ != hdr.dim[3]) {cout<<"dimensions do not agree."<<endl; return 1;}
	if (pixdimX != hdr.pixdim[1] || pixdimY != hdr.pixdim[2] || pixdimZ != hdr.pixdim[3]) {cout<<"pixdim do not agree."<<endl; return 1;}
	numSample = hdr.dim[4];
	//cout<<numSample;	
	array_input_f1 = new float[size_cube * numSample];
	if (array_input_f1 == NULL) {cout<<"Allocating memory failure."<<endl;	system("pause");	return 1;}
	filestream_input.read((char*)array_input_f1, sizeof(float) * size_cube * numSample);
	filestream_input.close();

	filestream_input.open(string_input_filename_f2.c_str(), ios::binary);		//f2
	if (!filestream_input.good()) {cout<<"Can't open\t"<<string_input_filename_f2.c_str()<<endl;	system("pause");	return 1;}
	filestream_input.read((char*) &hdr, HdrLen);
	if (dimX != hdr.dim[1] || dimY != hdr.dim[2] || dimZ != hdr.dim[3]) {cout<<"dimensions do not agree."<<endl; return 1;}
	if (pixdimX != hdr.pixdim[1] || pixdimY != hdr.pixdim[2] || pixdimZ != hdr.pixdim[3]) {cout<<"pixdim do not agree."<<endl; return 1;}
	array_input_f2 = new float[size_cube * numSample];
	if (array_input_f2 == NULL) {cout<<"Allocating memory failure."<<endl;	system("pause");	return 1;}
	filestream_input.read((char*)(array_input_f2), sizeof(float) * size_cube * numSample);
	filestream_input.close();

	filestream_input.open(string_input_filename_th1.c_str(), ios::binary);		//th1
	if (!filestream_input.good()) {cout<<"Can't open\t"<<string_input_filename_th1.c_str()<<endl;	system("pause");	return 1;}
	filestream_input.read((char*) &hdr, HdrLen);
	if (dimX != hdr.dim[1] || dimY != hdr.dim[2] || dimZ != hdr.dim[3]) {cout<<"dimensions do not agree."<<endl; return 1;}
	if (pixdimX != hdr.pixdim[1] || pixdimY != hdr.pixdim[2] || pixdimZ != hdr.pixdim[3]) {cout<<"pixdim do not agree."<<endl; return 1;}
	array_input_th1 = new float[size_cube * numSample];
	if (array_input_th1 == NULL) {cout<<"Allocating memory failure."<<endl;	system("pause");	return 1;}
	filestream_input.read((char*)array_input_th1, sizeof(float) * size_cube * numSample);
	filestream_input.close();

	filestream_input.open(string_input_filename_th2.c_str(), ios::binary);		//th2
	if (!filestream_input.good()) {cout<<"Can't open\t"<<string_input_filename_th2.c_str()<<endl;	system("pause");	return 1;}
	filestream_input.read((char*) &hdr, HdrLen);
	if (dimX != hdr.dim[1] || dimY != hdr.dim[2] || dimZ != hdr.dim[3]) {cout<<"dimensions do not agree."<<endl; return 1;}
	if (pixdimX != hdr.pixdim[1] || pixdimY != hdr.pixdim[2] || pixdimZ != hdr.pixdim[3]) {cout<<"pixdim do not agree."<<endl; return 1;}
	array_input_th2 = new float[size_cube * numSample];
	if (array_input_th2 == NULL) {cout<<"Allocating memory failure."<<endl;	system("pause");	return 1;}
	filestream_input.read((char*)(array_input_th2), sizeof(float) * size_cube * numSample);
	filestream_input.close();

	filestream_input.open(string_input_filename_ph1.c_str(), ios::binary);		//ph1
	if (!filestream_input.good()) {cout<<"Can't open\t"<<string_input_filename_ph1.c_str()<<endl;	system("pause");	return 1;}
	filestream_input.read((char*) &hdr, HdrLen);
	if (dimX != hdr.dim[1] || dimY != hdr.dim[2] || dimZ != hdr.dim[3]) {cout<<"dimensions do not agree."<<endl; return 1;}
	if (pixdimX != hdr.pixdim[1] || pixdimY != hdr.pixdim[2] || pixdimZ != hdr.pixdim[3]) {cout<<"pixdim do not agree."<<endl; return 1;}
	array_input_ph1 = new float[size_cube * numSample];
	if (array_input_ph1 == NULL) {cout<<"Allocating memory failure."<<endl;	system("pause");	return 1;}
	filestream_input.read((char*)array_input_ph1, sizeof(float) * size_cube * numSample);
	filestream_input.close();

	filestream_input.open(string_input_filename_ph2.c_str(), ios::binary);		//ph2
	if (!filestream_input.good()) {cout<<"Can't open\t"<<string_input_filename_ph2.c_str()<<endl;	system("pause");	return 1;}
	filestream_input.read((char*) &hdr, HdrLen);
	if (dimX != hdr.dim[1] || dimY != hdr.dim[2] || dimZ != hdr.dim[3]) {cout<<"dimensions do not agree."<<endl; return 1;}
	if (pixdimX != hdr.pixdim[1] || pixdimY != hdr.pixdim[2] || pixdimZ != hdr.pixdim[3]) {cout<<"pixdim do not agree."<<endl; return 1;}
	array_input_ph2 = new float[size_cube * numSample];
	if (array_input_ph2 == NULL) {cout<<"Allocating memory failure."<<endl;	system("pause");	return 1;}
	filestream_input.read((char*)(array_input_ph2), sizeof(float) * size_cube * numSample);
	filestream_input.close();

	float reciprocal_pixdimX = 1.0f/pixdimX;
	float reciprocal_pixdimY = 1.0f/pixdimY;
	float reciprocal_pixdimZ = 1.0f/pixdimZ;
	
	//make the macro definition if the image need to be reversed in X or Y directions
#ifdef FLIP_Y
	reverse_y(array_input_mask, 1, dimX, dimY, dimZ);
	#ifdef FINAL_SEED_TRACK
	reverse_y(array_input_seed, 1, dimX, dimY, dimZ);
	#endif
	reverse_y(array_input_f1, numSample, dimX, dimY, dimZ);
	reverse_y(array_input_f2, numSample, dimX, dimY, dimZ);
	reverse_y(array_input_th1, numSample, dimX, dimY, dimZ);
	reverse_y(array_input_th2, numSample, dimX, dimY, dimZ);
	reverse_y(array_input_ph1, numSample, dimX, dimY, dimZ);
	reverse_y(array_input_ph2, numSample, dimX, dimY, dimZ);
#endif
	
#ifdef FLIP_X
	reverse_x(array_input_mask, 1, dimX, dimY, dimZ);
	#ifdef FINAL_SEED_TRACK
	reverse_x(array_input_seed, 1, dimX, dimY, dimZ);
	#endif
	reverse_x(array_input_f1, numSample, dimX, dimY, dimZ);
	reverse_x(array_input_f2, numSample, dimX, dimY, dimZ);
	reverse_x(array_input_th1, numSample, dimX, dimY, dimZ);
	reverse_x(array_input_th2, numSample, dimX, dimY, dimZ);
	reverse_x(array_input_ph1, numSample, dimX, dimY, dimZ);
	reverse_x(array_input_ph2, numSample, dimX, dimY, dimZ);
#endif

	cout<<"Probabilistic (Multi-)Fiber Tracking on GPU..."<<endl
		<<"voxel dimension: "<<dimX<<"*"<<dimY<<"*"<<dimZ<<endl
		<<"voxel size: "<<pixdimX<<"*"<<pixdimY<<"*"<<pixdimZ<<endl
		<<"number of samples: "<<numSample<<endl
		<<"step_length: "<<para_step_length<<endl
#ifdef USE_F_THRESHOLD_AS_TERMINATE_CONDITION			//f1 or f2 or fa?
		<<"f threshold: "<<para_threshold_f<<endl
#endif
		<<"Will Output fibers, the number of whose steps larger than "<< min_step<<endl
		<<"=============================================="<<endl;
	// CPU initialize
	// ==============================================================================
	int num_seed_voxel = 0;
	
	#ifdef FINAL_SEED_TRACK
	//int num_seed_voxel = 0;
	for(int i=0;i<size_cube;i++)
		if(array_input_seed[i])
			num_seed_voxel++;
	//cout<<num_seed_voxel<<endl;
	#endif
	
	//A predefined number of tracking steps in each round with a parallel openCL device.
	//For details please refer to "Xu Mo, Probabilistic brain fiber tractography on gpus, IPDPSW, 2012"
	int iteration_time[ROUND_MAX] ={5,10,20,20,50,100,100,200,200,200,200,200,200};
	int iter_max = 0;
	for (int i = 0; i < ROUND_MAX; i ++)
	{
		
		if (iteration_time[i] > iter_max)
			iter_max = iteration_time[i];
	}

	numVoxel = 0;
#ifdef FINAL_SEED_TRACK
	numVoxel = num_seed_voxel;
#else
	for (int i = 0; i < size_cube; i++)
	{		
		if (array_input_mask[i] != 0)
			numVoxel ++;
	}
#endif

	// Everything is double because of the forward and backward directions
	result_sequence = new float[numVoxel * 8 * iter_max];

	step = new int[2 * numVoxel];						//
	iteration = new int[2 * numVoxel];					//
	length = new int[2 * numVoxel];						//
	reason = new int[2 * numVoxel];						//
	
	//idx2cub = new int[numVoxel];
	//cub2idx = new int[size_cube];
	start_coordinate = new float[8 * numVoxel];
	copy_start_coordinate = new float[8 * numVoxel];
	start_direction = new float[8 * numVoxel];
	int *map_back = new int [2 * numVoxel];
	for (int i = 0; i < 2 * numVoxel; i++)
	{
		map_back[i] = i;
	}

	cout<<numVoxel<<endl;
	/*
	numVoxel = 0;
	for (int i = 0; i < size_cube; i++)
	{
		if (array_input_mask[i] != 0)
		{
			idx2cub[numVoxel] = i;
			cub2idx[i] = numVoxel;
			
			numVoxel ++; 
		}
	}
	*/


	//////////////////////////////////////////////////////////////
	//                       Timer                              //
	CPU_preprocess_time = clock() - CPU_preprocess_time; 
	GPU_preprocess_time = clock(); 
	//                                                          //
	//////////////////////////////////////////////////////////////

	// Open CL initialize 
	// This part is just some initialization of opencl devices and arguments of CL kernels
	// ==============================================================================
	// Name of the file with the source code for the computation kernel
	const char* fiberTrack_cl = "./fiberTrack.cl";

    cl_uint numPlatforms = 0;
    cl_char platformName[100];
    
	//Get an OpenCL platform
    ciErr1 = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (ciErr1 != CL_SUCCESS)
        cout<<"Error in clGetPlatformID"<<endl;
    if (0 < numPlatforms) 
    {
        cl_platform_id *platforms = new cl_platform_id[numPlatforms];
        
        ciErr1 = clGetPlatformIDs(numPlatforms, platforms, NULL);
        for(unsigned int i = 0; i < numPlatforms; i++)
        {
            ciErr1 = clGetPlatformInfo(
                            platforms[i],
                            CL_PLATFORM_VENDOR,
                            sizeof(platformName),
                            platformName,
                            NULL);
            cpPlatform = platforms[i];
            if(!strcmp(
                    (const char*)platformName, 
                    "Advanced Micro Devices, Inc."))
                break;
        }
        cout<<"Platform Found: " << platformName<<"\n";
    }

	//Get the devices
    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
    if (ciErr1 != CL_SUCCESS)
        cout<<"Error in clGetDeviceIDs"<<endl;

    //Create the context
    cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErr1);
    if (ciErr1 != CL_SUCCESS)
        cout<<"Error in clCreateContext"<<endl;

    // Create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErr1);
    if (ciErr1 != CL_SUCCESS)
        cout<<"Error in clCreateCommandQueue"<<endl;

	//=============================================
    // Read the OpenCL kernel in from source file
	// Create the program
    // Build the program with 'mad' Optimization option
    // Create the kernel 
    std::string  sourceStr;
	#ifdef MAC
        char* flags = "-cl-fast-relaxed-math -DMAC";
    #else
        char* flags = "-cl-fast-relaxed-math";
    #endif

	sourceStr = convertToString(fiberTrack_cl);
	const char * source    = sourceStr.c_str();
    size_t sourceSize[]    = { strlen(source) };	
	cpProgram = clCreateProgramWithSource(cxGPUContext, 1, &source, sourceSize, &ciErr1);

	ciErr1 = clBuildProgram(cpProgram, 1, &cdDevice, NULL, NULL, NULL);
	if (ciErr1 != CL_SUCCESS)
    {
        size_t len;
        char buffer[8 * 1024];
         
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(cpProgram,cdDevice, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        system("pause");
    }
	

    if (ciErr1 != CL_SUCCESS)
	{    cout<<"Error in clBuildProgram"<<ciErr1<<endl; system("pause");}



    /*ciErr1 |= clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
    if (ciErr1 != CL_SUCCESS)
	{    cout<<"Error in clBuildProgram"<<ciErr1<<endl; system("pause");}*/
    ckKernel = clCreateKernel(cpProgram, "fiberTrack", &ciErr1);
    if (ciErr1 != CL_SUCCESS)
		cout<<"Error in clCreateKernel"<<endl;
	
	//=============================================
	// Creat buffers
	
	//kernel_step = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(float) * numVoxel, NULL, &ciErr1);
	kernel_iteration = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(int) * 2 * numVoxel, NULL, &ciErr1);
	//kernel_length = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(float) * numVoxel, NULL, &ciErr1);
	kernel_reason = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(int) * 2 * numVoxel, NULL, &ciErr1);
	kernel_start_coordinate = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float) * 8 * numVoxel, NULL, &ciErr1);
	kernel_start_direction = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float) * 8 * numVoxel, NULL, &ciErr1);

	if (ciErr1 != CL_SUCCESS)
				cout<<"Error in clCreateBuffer"<<endl;

	// Creat images
	image_format.image_channel_data_type = CL_FLOAT;
    image_format.image_channel_order = CL_INTENSITY;
	size_t origin[] = {0, 0, 0};
    size_t region[] = {dimX, dimY, dimZ};
	f1_image = clCreateImage3D(cxGPUContext, CL_MEM_READ_ONLY, &image_format, dimX, dimY, dimZ, 0, 0, NULL, &ciErr1);
	f2_image = clCreateImage3D(cxGPUContext, CL_MEM_READ_ONLY, &image_format, dimX, dimY, dimZ, 0, 0, NULL, &ciErr1);
	ph1_image = clCreateImage3D(cxGPUContext, CL_MEM_READ_ONLY, &image_format, dimX, dimY, dimZ, 0, 0, NULL, &ciErr1);
	ph2_image = clCreateImage3D(cxGPUContext, CL_MEM_READ_ONLY, &image_format, dimX, dimY, dimZ, 0, 0, NULL, &ciErr1);
	th1_image = clCreateImage3D(cxGPUContext, CL_MEM_READ_ONLY, &image_format, dimX, dimY, dimZ, 0, 0, NULL, &ciErr1);
	th2_image = clCreateImage3D(cxGPUContext, CL_MEM_READ_ONLY, &image_format, dimX, dimY, dimZ, 0, 0, NULL, &ciErr1);

	// Set the Argument values
	
	//ciErr1 |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&kernel_step);
	ciErr1 = clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&kernel_iteration);
	//ciErr1 |= clSetKernelArg(ckKernel, 3, sizeof(cl_mem), (void*)&kernel_length);
	ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&kernel_reason);
	ciErr1 |= clSetKernelArg(ckKernel, 3, sizeof(cl_mem), (void*)&f1_image);
	ciErr1 |= clSetKernelArg(ckKernel, 4, sizeof(cl_mem), (void*)&th1_image);
	ciErr1 |= clSetKernelArg(ckKernel, 5, sizeof(cl_mem), (void*)&ph1_image);
	ciErr1 |= clSetKernelArg(ckKernel, 6, sizeof(cl_mem), (void*)&f2_image);
	ciErr1 |= clSetKernelArg(ckKernel, 7, sizeof(cl_mem), (void*)&th2_image);
	ciErr1 |= clSetKernelArg(ckKernel, 8, sizeof(cl_mem), (void*)&ph2_image);
	
	
	//ciErr1 |= clSetKernelArg(ckKernel, 13, sizeof(int), (void*)&dimX);
	//ciErr1 |= clSetKernelArg(ckKernel, 14, sizeof(int), (void*)&dimY);
	//ciErr1 |= clSetKernelArg(ckKernel, 15, sizeof(int), (void*)&dimZ);
	ciErr1 |= clSetKernelArg(ckKernel, 13, sizeof(float), (void*)&para_step_length);
	//ciErr1 |= clSetKernelArg(ckKernel, 17, sizeof(float), (void*)&para_threshold_f);
	ciErr1 |= clSetKernelArg(ckKernel, 14, sizeof(float), (void*)&para_threshold_angular);
	//ciErr1 |= clSetKernelArg(ckKernel, 19, sizeof(int), (void*)&para_threshold_count);
	ciErr1 |= clSetKernelArg(ckKernel, 17, sizeof(int), (void*)&dimX);
	ciErr1 |= clSetKernelArg(ckKernel, 18, sizeof(int), (void*)&dimY);
	ciErr1 |= clSetKernelArg(ckKernel, 19, sizeof(int), (void*)&dimZ);

	ciErr1 |= clSetKernelArg(ckKernel, 20, sizeof(float), (void*) &reciprocal_pixdimX);
	ciErr1 |= clSetKernelArg(ckKernel, 21, sizeof(float), (void*) &reciprocal_pixdimY);
	ciErr1 |= clSetKernelArg(ckKernel, 22, sizeof(float), (void*) &reciprocal_pixdimZ);


	if (ciErr1 != CL_SUCCESS)
		cout<<"Error in clSetKernelArg"<<ciErr1<<endl; 
	
	//////////////////////////////////////////////////////////////
	//                       Timer                              //
	GPU_preprocess_time = clock() - GPU_preprocess_time; 
	process_time = clock(); 
	//                                                          //
	//////////////////////////////////////////////////////////////
	// =====================================
	// initialize finished
	
	// initialize statistics and start coodinates
	int reason_length = 0, reason_border = 0, reason_low_f = 0, reason_big_turn = 0, reason_unknown = 0, reason_begin_low_f = 0;
	int max_step = 0;
	int total_iteration_count = 0;
	int total_fiber_count = 0;
	
	int countVoxel = 0;
	for (int z = 0; z < dimZ; z++)
	{
		for (int y = 0; y < dimY; y++)
		{
			for (int x = 0; x < dimX; x++)
			{
				//if (array_input_mask[x + y * dimX + z * dimX * dimY] != 0)
				if (array_input_seed[x + y * dimX + z * dimX * dimY] != 0)
				{
					copy_start_coordinate[4 * countVoxel] = start_coordinate[4 * countVoxel] = (float)x ;
					copy_start_coordinate[4 * countVoxel + 1] = start_coordinate[4 * countVoxel + 1] = (float)y ;
					copy_start_coordinate[4 * countVoxel + 2] = start_coordinate[4 * countVoxel + 2] = (float)z ;
					copy_start_coordinate[4 * numVoxel + 4 * countVoxel] = start_coordinate[4 * numVoxel + 4 * countVoxel] = (float)x;
					copy_start_coordinate[4 * numVoxel + 4 * countVoxel + 1] = start_coordinate[4 * numVoxel + 4 * countVoxel + 1] = (float)y;
					copy_start_coordinate[4 * numVoxel + 4 * countVoxel + 2] = start_coordinate[4 * numVoxel + 4 * countVoxel + 2] = (float)z;
					countVoxel ++; 
				}
			}
		}
	}


	//initialize parameters used in a .trk file header
	char id_string[6];   
	id_string[0]='T';
    id_string[1]='R';
    id_string[2]='A';
    id_string[3]='C';
    id_string[4]='K';
    id_string[5]=0;

    short int dim[3];
	dim[0]=dimX;
    dim[1]=dimY;
    dim[2]=dimZ;
    
	float voxel_size[3];
	voxel_size[0]=pixdimX;
    voxel_size[1]=pixdimY;
    voxel_size[2]=pixdimZ;

    short int n_scalars=0;
	char scalar_name[10][20]={0};

    short int n_properties=0;
	char property_name[10][20]={0};
	
	//float vox_to_ras[4][4]={-2,0,0,78,0,1.9928,-0.1694,-79,0,0.1694,1.9928,-38,0,0,0,1};
	float vox_to_ras[4][4]={pixdimX,0,0,0,0,pixdimY,0,0,0,0,pixdimZ,0,0,0,0,1};

	char reserved[444]={0};

	char voxel_order[4]={76,65,83,0};   //???

	char pad2[4]={0};

	float image_orientation_patient[6]={1,0,0,0,-0.9964,0.0847};

	char pad1[2]={0};

    //unsigned char invert_x=0;
    //unsigned char invert_y=0;
    //unsigned char invert_z=0;
    //unsigned char swap_xy=0;
    //unsigned char swap_yz=0;
    //unsigned char swap_zx=0;
    
	int n_count=0;
    int version=2;
    int hdr_size=1000;
   
    origin[0]=0.0;
    origin[1]=0.0;
    origin[2]=0.0;

    char temp=0;
	
	int compare=0;
	    
	n_count=0;
		
	#ifdef FINAL_SEED_TRACK
	string fileOut5("final_seed.trk");
	#else
	string fileOut5("final_all.trk");
	#endif
	
	// Write these parameters into the .trk result file
	filestream_output.open(fileOut5,ios::binary);
	filestream_output.write(id_string,6);
	filestream_output.write((char *)dim,3*sizeof(short int));
	filestream_output.write((char *)voxel_size,3*sizeof(float));
	filestream_output.write((char *)origin,3*sizeof(float));
	filestream_output.write((char *)&n_scalars,sizeof(short int));
	filestream_output.write((char *)scalar_name,200);
	filestream_output.write((char *)&n_properties,sizeof(short int));
	filestream_output.write((char *)property_name,200);
	filestream_output.write((char *)vox_to_ras,16*sizeof(float));
	filestream_output.write((char *)reserved,444);
	filestream_output.write((char *)voxel_order,4);
	filestream_output.write((char *)pad2,4);
	filestream_output.write((char *)image_orientation_patient,6*sizeof(float));
	filestream_output.write((char *)pad1,2);
	filestream_output.write((char *)&temp,1);
	filestream_output.write((char *)&temp,1);
	filestream_output.write((char *)&temp,1);
	filestream_output.write((char *)&temp,1);
	filestream_output.write((char *)&temp,1);
	filestream_output.write((char *)&temp,1);
	filestream_output.write((char *)&n_count,sizeof(int));
	filestream_output.write((char *)&version,sizeof(int));
	filestream_output.write((char *)&hdr_size,sizeof(int));

	// =============================================
	// for each sample

    vector<vector<float>> track_result(2*numVoxel,vector<float>(0));
    vector<vector<float>> track_result1(2*numVoxel,vector<float>(0));
	vector<vector<float>> track_result2(2*numVoxel,vector<float>(0));

	srand((unsigned)time(NULL));
	float *array_input_f1_new = new float[size_cube];
	float *array_input_f2_new = new float[size_cube];
	float *array_input_ph1_new = new float[size_cube];
	float *array_input_ph2_new = new float[size_cube];
	float *array_input_th1_new = new float[size_cube];
	float *array_input_th2_new = new float[size_cube];
	
	#ifdef FINAL_SEED_TRACK
	char *flag=new char[2*numVoxel];
	#endif

	/*int ***record;
	record= new int**[dimZ];
	for (int i=0;i<dimZ;i++)
	{
		record[i]=new int*[dimY];
		for(int j=0;j<dimY;j++)
		{
		 	record[i][j]=new int[dimX];
		    memset((void *) record[i][j], 0, sizeof(int)*dimX );
		}
	}*/
	
	//int *probmap = new int [size_cube];
	//memset((void *) probmap, 0, sizeof(int)*size_cube);		

	int count_r = 0;
	
	for(int sample = 0; sample < Num_Probtrack_Sample; sample++)
	{
		//multiple samples, randomly choose one direction sample
		//when tracking to a specific voxel
		for (int i=0;i<size_cube;i++)
		{
			 int num=randn(numSample);
			 array_input_f1_new[i]=array_input_f1[num*size_cube+i];
			 array_input_f2_new[i]=array_input_f2[num*size_cube+i];
			 array_input_ph1_new[i]=array_input_ph1[num*size_cube+i];
			 array_input_th1_new[i]=array_input_th1[num*size_cube+i];
			 array_input_ph2_new[i]=array_input_ph2[num*size_cube+i];
			 array_input_th2_new[i]=array_input_th2[num*size_cube+i];
		}	
	
	    //double direction tracking
		#ifdef FINAL_SEED_TRACK
		memset(flag,0,sizeof(char)*2*numVoxel);
		#endif
		
	
		for (int turn=0;turn<2;turn++)
		{	
			//if(turn==0)
				//cout<<"forward:"<<endl;
			//else
				//cout<<"backward:"<<endl;

			if(turn==1)
			{
				for(int i=0;i<size_cube;i++)
				{
					array_input_ph1_new[i]=-array_input_ph1_new[i];
				    array_input_ph2_new[i]=-array_input_ph2_new[i];
					array_input_th1_new[i]=-array_input_th1_new[i];
					array_input_th2_new[i]=-array_input_th2_new[i];
				}
			}
		
			numValidVoxel = 2 * numVoxel;
			for (int n = 0; n < numVoxel; n++)
			{
				start_coordinate[4 * n] = copy_start_coordinate[4 * n];            //X axis
				start_coordinate[4 * n + 1] = copy_start_coordinate[4 * n + 1];    //Y axis
				start_coordinate[4 * n + 2] = copy_start_coordinate[4 * n + 2];    //Z axis
				start_coordinate[4 * numVoxel + 4 * n] = copy_start_coordinate[4 * numVoxel + 4 * n];
				start_coordinate[4 * numVoxel + 4 * n + 1] = copy_start_coordinate[4 * numVoxel + 4 * n + 1];
				start_coordinate[4 * numVoxel + 4 * n + 2] = copy_start_coordinate[4 * numVoxel + 4 * n + 2];
				length[n] = 0;
				length[n + numVoxel] = 0;
				map_back[n] = n;
				map_back[n + numVoxel] = n + numVoxel;
			}
			//////////////////////////////////////////
			//                 Timer                //
			GPU_transfer_time = clock(); 
			//                                      //
			//////////////////////////////////////////
			// Write images
	
			ciErr1 = clEnqueueWriteImage(cqCommandQueue, f1_image, CL_TRUE, origin, region, 0, 0, array_input_f1_new, 0,  0, 0);
			ciErr1 |= clEnqueueWriteImage(cqCommandQueue, ph1_image, CL_TRUE, origin, region, 0, 0, array_input_ph1_new, 0, 0, 0);
			ciErr1 |= clEnqueueWriteImage(cqCommandQueue, th1_image, CL_TRUE, origin, region, 0, 0, array_input_th1_new, 0, 0, 0);
			ciErr1 |= clEnqueueWriteImage(cqCommandQueue, f2_image, CL_TRUE, origin, region, 0, 0, array_input_f2_new, 0,  0, 0);
			ciErr1 |= clEnqueueWriteImage(cqCommandQueue, ph2_image, CL_TRUE, origin, region, 0, 0, array_input_ph2_new, 0, 0, 0);
			ciErr1 |= clEnqueueWriteImage(cqCommandQueue, th2_image, CL_TRUE, origin, region, 0, 0, array_input_th2_new, 0, 0, 0);
			if (ciErr1 != CL_SUCCESS)
				cout<<"Error in clEnqueueWriteImage"<<ciErr1<<endl;
			//////////////////////////////////////////
			//                 Timer                //
			GPU_transfer_time = clock() - GPU_transfer_time; 
			total_GPU_transfer_time += GPU_transfer_time;
			//                                      //
			//////////////////////////////////////////


			//*********************************************************//
			//                        debug                            //
			cl_mem cl_debug = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(float) * numValidVoxel, NULL, &ciErr1);
			ciErr1 = clSetKernelArg(ckKernel, 16, sizeof(cl_mem), (void*)&cl_debug);
			//                                                         //
			//*********************************************************//
			
			


		
			int result_count=0;
			int iter_count=0;
			int position=0;
			for (int round = 0; round < ROUND_MAX && numValidVoxel != 0; round++)
			{
				cout << "round : "<<round<<endl;
				//cout<<"Sample: "<<"multi_rand"<<"\t"
					//<<"Round: "<<round<<"\t"
					//<<"# Thread: "<<numValidVoxel<<endl;
				//int round = 0;
				//////////////////////////////////////////
				//                 Timer                //
				GPU_transfer_time = clock(); 
				//                                      //
				//////////////////////////////////////////
				// ====================================
				// GPU process	
				if (kernel_result) clReleaseMemObject(kernel_result);
				kernel_result = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(float) * numValidVoxel * 4 * iteration_time[round], NULL, &ciErr1);
				//kernel_result = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(float) * numValidVoxel * 4, NULL, &ciErr1);
				if (ciErr1 != CL_SUCCESS)
					cout<<"Error in clCreateBuffer"<<ciErr1<<endl;
				// Write Buffers
				ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, kernel_start_coordinate, CL_TRUE, 0, sizeof(float) * 4 * numValidVoxel, start_coordinate, 0, NULL, NULL);
				if (round >= 1)
					ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, kernel_start_direction, CL_TRUE, 0, sizeof(float) * 4 * numValidVoxel, start_direction, 0, NULL, NULL);
		
				if (ciErr1 != CL_SUCCESS)
					cout<<"Error in clEnqueueWriteBuffer"<<ciErr1<<endl;
		
				// Set the Argument values
				ciErr1 = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&kernel_result);
				ciErr1 |= clSetKernelArg(ckKernel, 9, sizeof(cl_mem), (void*)&kernel_start_coordinate);
				ciErr1 |= clSetKernelArg(ckKernel, 10, sizeof(cl_mem), (void*)&kernel_start_direction);
				ciErr1 |= clSetKernelArg(ckKernel, 11, sizeof(int), (void*)&round);
				ciErr1 |= clSetKernelArg(ckKernel, 12, sizeof(int), (void*)(iteration_time+round));
				ciErr1 |= clSetKernelArg(ckKernel, 15, sizeof(int), (void*)&numValidVoxel);
				if (ciErr1 != CL_SUCCESS)
				cout<<"Error in clSetKernelArg"<<ciErr1<<endl; 
		
				// set and log Global and Local work size dimensions
				szLocalWorkSize = 256;
				szGlobalWorkSize = (numValidVoxel + szLocalWorkSize - 1) / szLocalWorkSize * szLocalWorkSize;
				//szGlobalWorkSize  = szLocalWorkSize;

				//////////////////////////////////////////
				//                 Timer                //
				GPU_transfer_time = clock() - GPU_transfer_time; 
				total_GPU_transfer_time += GPU_transfer_time;
				GPU_kernel_time = clock();
				//                                      //
				//////////////////////////////////////////
	
				/*********************************************************/
				// run the kernel (The main tracking process is here), get the result_sequence
				ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, &ceEvent);    
				if (ciErr1 != CL_SUCCESS)
					cout<<"Error in clEnqueueNDRangeKernel"<<ciErr1<<endl;				
				clWaitForEvents(1, &ceEvent);
				//////////////////////////////////////////
				//                 Timer                //
				GPU_kernel_time = clock() - GPU_kernel_time; 
				//cout<<"kernel time: "<<GPU_kernel_time<<" ms."<<endl;
				total_GPU_kernel_time += GPU_kernel_time;
				GPU_transfer_time = clock(); 
				//                                      //
				//////////////////////////////////////////
				//write the tracking path (kernel_result) to result_sequence
				ciErr1 = clEnqueueReadBuffer (cqCommandQueue, kernel_result, CL_TRUE, 0, sizeof(float) * 4 * numValidVoxel * iteration_time[round], result_sequence, 0, NULL, NULL);
				//ciErr1 = clEnqueueReadBuffer (cqCommandQueue, kernel_step, CL_TRUE, 0, sizeof(int) * numValidVoxel, step, 0, NULL, NULL);
				ciErr1 = clEnqueueReadBuffer (cqCommandQueue, kernel_iteration, CL_TRUE, 0, sizeof(int) * numValidVoxel, iteration, 0, NULL, NULL);
				//ciErr1 = clEnqueueReadBuffer (cqCommandQueue, kernel_length, CL_TRUE, 0, sizeof(float) * numValidVoxel, length, 0, NULL, NULL);
				ciErr1 = clEnqueueReadBuffer (cqCommandQueue, kernel_reason, CL_TRUE, 0, sizeof(int) * numValidVoxel, reason, 0, NULL, NULL);

				ciErr1 = clEnqueueReadBuffer (cqCommandQueue, kernel_start_coordinate, CL_TRUE, 0, sizeof(float) * 4 * numValidVoxel, start_coordinate, 0, NULL, NULL);
				ciErr1 = clEnqueueReadBuffer (cqCommandQueue, kernel_start_direction, CL_TRUE, 0, sizeof(float) * 4 * numValidVoxel, start_direction, 0, NULL, NULL);
				if (ciErr1 != CL_SUCCESS)
					cout<<"Error in clEnqueueReadBuffer"<<endl;
				/*********************************************************/
	
				//////////////////////////////////////////
				//                 Timer                //
				GPU_transfer_time = clock() - GPU_transfer_time; 
				total_GPU_transfer_time += GPU_transfer_time;
				postprocess_time = clock(); 
				//                                      //
				//////////////////////////////////////////
				//record length
	
	
				// cout result_sequence///////////////////
				/*string fileOut4("result_sequence");
				char tmp[10];
			    itoa(round,tmp,10);
				fileOut4.append(tmp);
			    filestream_output.open(fileOut4,ios::binary);
			    filestream_output.write((char *)result_sequence,10*8*numVoxel*sizeof(float));
				filestream_output.close();
				*/
				///////////////////////////////////////////
				
				//process the result sequence to track_result (forward tracking path) 
				//and track_result1 (backward tracking path)
			    int seek=0;
				if(turn==0)
				{
				    double sum;
				    result_count=0;
				    for (int i=iter_count;i<iter_count+iteration_time[round];i++)
				    {
					    for (int j=0;j<numValidVoxel;j++)
					    {
						    if(round==0)
						    {
							    sum=0.0;
							    for (int k=0;k<3;k++)
							    {
								    sum+=result_sequence[result_count];
								    result_count++;
							    }
							    result_count=result_count-3;
							    //if sum is 0, the tracking path already terminate. 
								if(sum!=0)
							    {
							        for (int k=0;k<3;k++)
						            {	
								        track_result[j].push_back(result_sequence[result_count]); 	
								        result_count++;
							        }	
									#ifdef FINAL_SEED_TRACK
									seek=floor(result_sequence[result_count-3])+dimX*floor(result_sequence[result_count-2])+dimX*dimY*floor(result_sequence[result_count-1]);
									if( array_input_seed[seek] && result_sequence[result_count-3]>0&&result_sequence[result_count-2]>0&&result_sequence[result_count-1]>0)    
										flag[j]=1;
									#endif
							        result_count++;
							    }
							    else
								    result_count+=4;
						    }
						    else
						    {
							    sum=0.0;
							    for (int k=0;k<3;k++)
							    {
								    sum+=result_sequence[result_count];
								    result_count++;
							    }
							    result_count=result_count-3;
							    if(sum!=0)
							    {
							        for (int k=0;k<3;k++)
						            {	
								        track_result[map_back[j]].push_back(result_sequence[result_count]); 
								        result_count++;
							        }
									#ifdef FINAL_SEED_TRACK
									seek=floor(result_sequence[result_count-3])+dimX*floor(result_sequence[result_count-2])+dimX*dimY*floor(result_sequence[result_count-1]);
									if(array_input_seed[seek] && result_sequence[result_count-3]>0&&result_sequence[result_count-2]>0&&result_sequence[result_count-1]>0)                
										flag[map_back[j]]=1;
									#endif
							        result_count++;
							    }
							    else
								    result_count+=4;
						    }
					    }
				    }
				    iter_count=iter_count+iteration_time[round];
				}
				else if(turn==1)
				{
					double sum;
				    result_count=0;
				    for (int i=iter_count;i<iter_count+iteration_time[round];i++)
				    {
					    for (int j=0;j<numValidVoxel;j++)
					    {
							if(round==0)
							{
								sum=0.0;
								for (int k=0;k<3;k++)
								{
									sum+=result_sequence[result_count];
									result_count++;
								}
								result_count=result_count-3;
								if(sum!=0)
								{
								    for (int k=0;k<3;k++)
							        {	
									    track_result1[j].push_back(result_sequence[result_count]); 	
									    result_count++;
								    }		
									#ifdef FINAL_SEED_TRACK
									seek=floor(result_sequence[result_count-3])+dimX*floor(result_sequence[result_count-2])+dimX*dimY*floor(result_sequence[result_count-1]);
									if( array_input_seed[seek] && result_sequence[result_count-3]>0&&result_sequence[result_count-2]>0&&result_sequence[result_count-1]>0)
										flag[j]=1;
									#endif
								    result_count++;
								}
								else
									result_count+=4;
							 }
							 else
							 {
								sum=0.0;
								for (int k=0;k<3;k++)
								{
									sum+=result_sequence[result_count];
									result_count++;
								}
								result_count=result_count-3;
								if(sum!=0)
								{
								    for (int k=0;k<3;k++)
							        {	
									    track_result1[map_back[j]].push_back(result_sequence[result_count]); 
									    result_count++;
								    }

									#ifdef FINAL_SEED_TRACK
									seek=floor(result_sequence[result_count-3])+dimX*floor(result_sequence[result_count-2])+dimX*dimY*floor(result_sequence[result_count-1]);
									if( array_input_seed[seek] && result_sequence[result_count-3]>0&&result_sequence[result_count-2]>0&&result_sequence[result_count-1]>0)
										flag[map_back[j]]=1;
									#endif
								    
									result_count++;
								}
								else
									result_count+=4;
							  }
							}
						}
					    iter_count=iter_count+iteration_time[round];
					}

				for (int v = 0; v < numValidVoxel; v++)
				{
					length[map_back[v]] += iteration[v];
				}

				//compact
				int pos = 0;
				for (int v = 0; v < numValidVoxel; v++)
				{
					if (iteration[v] == iteration_time[round])
					{
						start_coordinate[4 * pos] = start_coordinate[4 * v];
						start_coordinate[4 * pos + 1] = start_coordinate[4 * v + 1];
						start_coordinate[4 * pos + 2] = start_coordinate[4 * v + 2];
						start_direction[4 * pos] = start_direction[4 * v];
						start_direction[4 * pos + 1] = start_direction[4 * v + 1];
						start_direction[4 * pos + 2] = start_direction[4 * v + 2];
						map_back[pos] = map_back[v];
						pos ++;
					}
				}
				numValidVoxel = pos;
			
				//////////////////////////////////////////
				//                 Timer                //
				postprocess_time = clock() - postprocess_time; 
				total_postprocess_time += postprocess_time;
				//cout<<i<<"th sample: CPU time: "<<CPU_time<<" ms."<<endl;
				//                                      //
				//////////////////////////////////////////
	
				//*********************************************************//
				//                        debug                            //
				//float *debug = new float[numValidVoxel];
				//ciErr1 = clEnqueueReadBuffer (cqCommandQueue, cl_debug, CL_TRUE, 0, sizeof(float) * numValidVoxel, debug, 0, NULL, NULL);
				//if (round == 1)
				//{
				//	for (int outer = 0; outer < 12; outer ++)
				//	{
				//		for (int inner = 0; inner < 7; inner ++)
				//		{
				//			cout<<debug[outer * 7 + inner]<<endl;
				//		}
				//		cout <<endl;
					//	}
				//}
	
			}
		    }
	
			
			/*for (int i=0;i<iter_count/4;i++)
				{
					cout<<track_result[0][i*4]<<' '<<track_result[0][i*4+1]<<' '<<track_result[0][i*4+2]<<' '<<track_result[0][i*4+3]<<endl;
				}*/
					
	
			//*********************************************************//
			//                        debug                            //
			//float *debug = new float[numValidVoxel];
			//ciErr1 = clEnqueueReadBuffer (cqCommandQueue, cl_debug, CL_TRUE, 0, sizeof(float) * numValidVoxel, debug, 0, NULL, NULL);
			/*for (int outer = 0; outer < 120; outer ++)
			{
				for (int inner = 0; inner < 10; inner ++)
				{
					cout<<debug[outer * 10 + inner]<<"\t";
				}
				cout <<endl;
			}*/
			//cout<<step[24815]<<endl;
			//cout<<length[24815]<<endl;
			//                                                         //
			//*********************************************************//		
			
			
			// ====================================
			// CPU postprocess
			//////////////////////////////////////////
			//                 Timer                //
			postprocess_time = clock(); 
			//                                      //
			//////////////////////////////////////////
	
			
			//cout<<numVoxel<<endl;	
			/*string fileOut("length_");
			char tmp[10];
			itoa(i,tmp,10);
			fileOut.append(tmp);
			filestream_output.open(fileOut, ios::binary);
			filestream_output.write((char*)length, sizeof(int) * 2 * numVoxel);
			filestream_output.close();*/
	
			/*string fileOut1("result_");
			char tmp1[10];
			itoa(i,tmp1,10);
			fileOut1.append(tmp1);
			filestream_output.open(fileOut1, ios::binary);
			filestream_output.write((char*)result_sequence, sizeof(float) * numVoxel * 8 * iter_max);
			filestream_output.close();
			*/
		
			/*float **track_result_3=NULL;
			track_result_3=new float *[2*numVoxel];
			for (int i=0;i<2*numVoxel;i++)
				track_result_3[i]=new float[3*1000];

			for (int i=0;i<2*numVoxel;i++)
			for (int j=0;j<1000*3;j++)
					track_result_3[i][j]=0.0;
			*/

			
			// integrate the forward and backward tracking results (track_result and track_result1) into track_result2
			for (int i=0;i<2*numVoxel;i++)
			{
				int len2=track_result1[i].size()/3;
				for (int j=len2-1;j>0;j--)
				{
					for (int k=0;k<3;k++)
						track_result2[i].push_back(track_result1[i][(j-1)*3+k]);
				}
				for (int j=3;j<track_result[i].size();j++)
				{
					track_result2[i].push_back(track_result[i][j]);
				}
		 	}



		/*
		int tempf=0;
		int tempc=0;
		for (int i=0;i<5;i++)
		{
			cout<<track_result2[i].size()<<' ';
			for (int j=0;j<track_result2[i].size();j++)
				cout<<track_result2[i][j]<<' ';
			cout<<endl;
		}*/
        int temp2;
		int seek1=0;
		int count_prob=0;

		coor_idx coor_temp_old = {-1,-1,-1};
		coor_idx coor_temp_current = {0,0,0};

#ifdef FINAL_SEED_TRACK
		for(int i=0;i<2*numVoxel;i++)
		{
			temp2= (int)(track_result2[i].size()/3);
			count_prob=0;

			if(temp2>=Fiber_Length_Min && flag[i]==1)
			{
				if (temp2>compare)
					compare=temp2;
				//cout<<temp2<<endl;
				//bool longdist = FALSE;
				//coor_temp_old.x = coor_temp_current.x = floor(track_result2[i][0]);
				//coor_temp_old.y = coor_temp_current.y = floor(track_result2[i][1]);
				//coor_temp_old.z = coor_temp_current.z = floor(track_result2[i][2]); 

				for(int j=0;j<3*temp2;j++)
				{	
					if(j%3==2)
					{
						coor_temp_current.x = floor(track_result2[i][j-2]);
						coor_temp_current.y = floor(track_result2[i][j-1]);
						coor_temp_current.z = floor(track_result2[i][j]);
						seek1=coor_temp_current.x+coor_temp_current.y*dimX+coor_temp_current.z*dimX*dimY;
						
						#ifdef USE_PROBMASK
						if(array_input_probmask[seek1]>=threshold_prob)
						    count_prob++;
						#endif

						if (same_voxel(coor_temp_current,coor_temp_old))
							continue;
						//longdist = longdist_coor (coor_temp_current,coor_temp_old);
						//if (longdist)
						//	cout << coor_temp_current.x <<' '<< coor_temp_current.y <<' '<< coor_temp_current.z<<endl
						//		 << coor_temp_old.x <<' '<< coor_temp_old.y <<' '<< coor_temp_old.z<<endl;
						coor_temp_old = coor_temp_current;
						//probmap[coor2index(coor_temp_current,dimX,dimY)]++;
						
					//	record[x_r][y_r][z_r]++;
					}
				}
				//if (longdist)
				//	continue;

				bool save_this_fiber = TRUE;
				#ifdef USE_PROBMASK
					save_this_fiber =  (float) count_prob / temp2 >= threshold;
				#endif

				if( save_this_fiber )
				{
					filestream_output.write((char *)&temp2,sizeof(int));
				    for( int k=0;k<3*temp2;k++)
				    {
					    float temp3;
					    temp3=track_result2[i][k] * voxel_size[k%3];
					    filestream_output.write((char *)&temp3,sizeof(float));
			     	}
					n_count++;
				}
			
			}
			
		}
#else		
		for(int i=0;i<2*numVoxel;i++)
		{
			temp2= (int)(track_result2[i].size()/3);
			count_prob=0;



			if(temp2>=Fiber_Length_Min)
			{
				if (temp2>compare)
					compare=temp2;
								
				filestream_output.write((char *)&temp2,sizeof(int));
				for( int k=0;k<3*temp2;k++)
				{
				    float temp3;
					temp3=track_result2[i][k] * voxel_size[k%3];
				    filestream_output.write((char *)&temp3,sizeof(float));
			    }
				n_count++;
			}
			
		}
#endif
		
		cout<<"max_length"<<compare<<endl;
		
		for(int i=0;i<2*numVoxel;i++)
		{
			track_result[i].clear();
			track_result1[i].clear();
			track_result2[i].clear();
		}

		cout<<"sample "<<sample<<" tracking done!"<<endl;
				
		/*string fileOut2("track_result");
		filestream_output.open(fileOut2,ios::binary);
		//for(int i=0;i<numVoxel/2;i++)
		//{
	    filestream_output.write((char *)track_result_3[10000], sizeof(float) * 3 * iter_max);
		//}
		filestream_output.close();*/
 
	}


	filestream_output.close();


	/*
	 string fileOut6="record_region";
	 filestream_output.open(fileOut6,ios::binary);
	 filestream_output.write((char *)record,sizeof(int)*dimX*dimY*dimZ);   	
	 filestream_output.close();
	 
	 string fileOut6="probmap.nii";
	 hdr_mask[35] = 8;
	 hdr_mask[36] = 32;
	 filestream_output.open(fileOut6,ios::binary);
	 filestream_output.write((char *)hdr_mask, HdrLen);
	 filestream_output.write((char *)probmap, sizeof(int)*size_cube);   	
	 filestream_output.close();
	 */
	 
	 //***************************************************************************************************
		//fileOut = string("iteration_");
		//fileOut.append(tmp);
		//filestream_output.open(fileOut, ios::binary);
		//filestream_output.write((char*)iteration, sizeof(int) * numValidVoxel);
		//filestream_output.close();
		//
		//fileOut = string("idx_valid2cub");
		//
		//itoa(i,tmp,10);
		//fileOut.append(tmp);
		//filestream_output.open(fileOut, ios::binary);
		//filestream_output.write((char*)idx_valid2cub, sizeof(int) * numValidVoxel);
		//filestream_output.close();
		//fileOut = string("cub2idx_valid");
		//
		//itoa(i,tmp,10);
		//fileOut.append(tmp);
		//filestream_output.open(fileOut, ios::binary);
		//filestream_output.write((char*)cub2idx_valid, sizeof(int) * size_cube);
		//filestream_output.close();

		//int * iteration_cube = new int[size_cube];
		//for (int u = 0; u < size_cube; u++)
		//{
		//	iteration_cube[u] = 0;
		//}
		//for (int u = 0; u < numValidVoxel; u++)
		//{
		//	iteration_cube[idx_valid2cub[u]] = iteration[u];
		//}
		//
		//fileOut = string("iteration_cube_");
		//fileOut.append(tmp);
		//filestream_output.open(fileOut, ios::binary);
		//filestream_output.write((char*)iteration_cube, sizeof(int) * size_cube);
		//filestream_output.close();
		//delete[]iteration_cube;
		//for (int t = 0; t < numValidVoxel; t++)
		//{
		//	switch (reason[i])
		//	{
		//		case 1:
		//			reason_length++;
		//			break;
		//		case 2:
		//			reason_border++;
		//			break;
		//		case 3:
		//			reason_low_f++;
		//			break;
		//		case 4:
		//			reason_big_turn++;
		//			break;
		//		case 5:
		//			reason_begin_low_f++;
		//			break;
		//		default:
		//			reason_unknown ++;
		//	}
		//	total_iteration_count += iteration[t];
		//	if (step[t] > max_step)
		//		max_step = step[t];
		//	if (step[t] > 75)
		//	{
		//		int index = idx_valid2cub[t];
		//		int z = index / (dimX * dimY);
		//		int y = (index - z * (dimX * dimY)) / dimX;
		//		int x = (index - z * (dimX * dimY) - y * dimX);
		//		cout
		//			//<<"Voxel: "<<cub2idx[idx_valid2cub[t]]
		//			<<"seed: "
		//			<<x<<" "<<y<<" "<<z
		//			<<"  Sample: "<<i
		//			//<<"\tValidIndex"<<t
		//			<<"  step: "<<step[t]
		//			<<"  iteration: "<<iteration[t]
		//			<<"  length: "<<length[t]
		//			//<<"  Reason: "<<reason[t]
		//			<<endl;
		//		total_fiber_count ++;
				// =============================================
			
				
				//short *out_cube = new short[size_cube];
				//for (int u = 0; u < size_cube; u++)
				//	out_cube[u] = 0;
				//for (int u = 0; u < step[t]; u++)
				//{
				//	out_cube[result_sequence[t + u * numValidVoxel]] = 1;
				//	//cout<<debug[t + u * numValidVoxel]<<endl;
				//}
				//string fileOut("prob_track_");
				//char tmp[10];
				//itoa(x,tmp,10);
				//fileOut.append(tmp).append("_");
				//itoa(y,tmp,10);
				//fileOut.append(tmp).append("_");
				//itoa(z,tmp,10);
				//fileOut.append(tmp).append("_");
				//itoa(i,tmp,10);
				//fileOut.append(tmp).append(".nii");
				//filestream_output.open(fileOut, ios::binary);
				//filestream_output.write((char*)hdr_mask, sizeof(char) * HdrLen);
				//filestream_output.write((char*)out_cube, sizeof(short) * size_cube);
				//filestream_output.close();
				//delete []out_cube;
		//	}
		//}
		//////////////////////////////////////////
		//                 Timer                //
		postprocess_time = clock() - postprocess_time; 
		total_postprocess_time += postprocess_time;
		//                                      //
		//////////////////////////////////////////
		
	//}
	//////////////////////////////////////////////////////////////
	//                       Timer                              //
	process_time = clock() - process_time; 
	postprocess_time = clock();
	//                                                          //
	//////////////////////////////////////////////////////////////
	// ====================================
	// write files... 

	//filestream_output.open(string_output_filename_sequence_merge, ios::binary);
	//if (!filestream_output.good()) {cout<<"Can't open\t"<<string_output_filename_sequence_merge<<endl;	system("pause");	return 1;}
	//filestream_output.close();
	//
	//filestream_output.open(string_output_filename_merge_nii, ios::binary);
	//if (!filestream_output.good()) {cout<<"Can't open\t"<<string_output_filename_merge_nii<<endl;	system("pause");	return 1;}
	//filestream_output.close();

	//filestream_output.open(string_output_filename_nii, ios::binary);
	//if (!filestream_output.good()) {cout<<"Can't open\t"<<string_output_filename_nii<<endl;	system("pause");	return 1;}
	//filestream_output.close();
	//
	//filestream_output.open(string_output_filename_info, ios::binary);
	//if (!filestream_output.good()) {cout<<"Can't open\t"<<string_output_filename_info<<endl;	system("pause");	return 1;}
	//filestream_output.close();

	//fout_count.write((char*)out_fiber_count, sizeof(int) * numSample * numVoxel);
	//fout_count.close();

	//fout.write((char*)hdr_mask, sizeof(char) * HdrLen);
	//fout.write((char*)output, sizeof(short) * size_cube);
	//fout.close();

	//////////////////////////////////////////////////////////////
	//                       Timer                              //
	postprocess_time = clock() - process_time; 
	total_postprocess_time += postprocess_time;
	//                                                          //
	//////////////////////////////////////////////////////////////

	cout<<"total fiber count: "<<n_count<<endl;

	cout<<"==================================================="<<endl<<"Congratulations!"<<endl
		<<"Read File and CPU preprocess time: "<<1.0*CPU_preprocess_time/1000<<" seconds."<<endl
		<<"Write Buffer and GPU preprocess time: "<<1.0*GPU_preprocess_time/1000<<" seconds."<<endl
		<<"Total GPU kernel time: "<<1.0*total_GPU_kernel_time/1000<<" seconds."<<endl
		<<"Total GPU transfer time: "<<1.0*total_GPU_transfer_time/1000<<" seconds."<<endl
		<<"Total CPU postprocess time: "<<1.0*total_postprocess_time/1000<<" seconds."<<endl
		<<"Total Iteration count: "<<total_iteration_count<<endl
		<<"fiber count (step>"<<min_step<<"): "<<total_fiber_count<<endl
		<<"max step: "<<max_step<<endl
		<<endl;

	cout<<"==================================================="<<endl
		<<"Stop Reason Statistics: "<<endl
		<<"Too Long: "<<reason_length<<endl
		<<"Go to Border: "<<reason_border<<endl
		<<"Low anisotropy: "<<reason_low_f<<endl
		<<"Sharp Turn: "<<reason_big_turn<<endl
		<<"Unknown: "<<reason_unknown<<endl;

	// =============================================
	// Free host memory
	delete []array_input_mask;
	delete []array_input_f1;
	delete []array_input_th1;
	delete []array_input_ph1;
	delete []array_input_f2;
	delete []array_input_th2;
	delete []array_input_ph2;
	//delete []result_sequence_merge;
	//delete []array_output_step;
	//delete []array_output_length;
	//delete []array_output_reason;
	delete []result_sequence;
	//delete []step;
	delete []iteration;
	//delete []length;
	delete []reason;
	//delete []seed_direction;
	//delete []idx2cub;
	//delete []cub2idx;
	delete []start_coordinate;
	delete []start_direction;
	delete []copy_start_coordinate;
	delete []map_back;
	
	/*
	for (int i=0;i<dimZ;i++)
	{
		for(int j=0;j<dimY;j++)
		{
		 	delete [](record[i][j]);
		}
		delete []record[i];
	}
	*/
	//delete []probmap;
	//delete []idx_Seed2cub;
	//delete []idx_valid2cub;


	Cleanup(0);
	
	system("pause");
   

	return 0;	
}


void Cleanup (int iExitCode)
{
    // Cleanup allocated objects
    if(cPathAndName)free(cPathAndName);
    if(cSourceCL)free(cSourceCL);

	if(ckKernel)clReleaseKernel(ckKernel);  
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);


	if(f1_image) clReleaseMemObject(f1_image);
	if(ph1_image) clReleaseMemObject(ph1_image);
	if(th1_image) clReleaseMemObject(th1_image);
	if(f2_image) clReleaseMemObject(f2_image);
	if(ph2_image) clReleaseMemObject(ph2_image);
	if(th2_image) clReleaseMemObject(th2_image);

	if(kernel_result) clReleaseMemObject(kernel_result);
	if(kernel_start_coordinate) clReleaseMemObject(kernel_start_coordinate);
	if(kernel_start_direction) clReleaseMemObject(kernel_start_direction);
	if(kernel_iteration) clReleaseMemObject(kernel_iteration);
	if(kernel_reason) clReleaseMemObject(kernel_reason);

    // finalize logs and leave
    //exit (iExitCode);
}


string convertToString(const char *filename)
{
	size_t size;
	char*  str;
	std::string s;

	std::fstream f(filename, (std::fstream::in | std::fstream::binary));

	if(f.is_open())
	{
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);

		str = new char[size+1];
		if(!str)
		{
			f.close();
			return NULL;
		}

		f.read(str, fileSize);
		f.close();
		str[size] = '\0';
	
		s = str;
		delete[] str;
		return s;
	}
	else
	{
		std::cout << "\nFile containg the kernel code(\".cl\") not found. Please copy the required file in the folder containg the executable.\n";
		exit(1);
	}
	return NULL;
}
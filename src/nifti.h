typedef struct _nii_hdr
{             /* NIFTI-1 usage      */    /*ANALYZE 7.5 field(s)*/     /* Byte offset */
                      /**********************/    /**********************/     /***************/


 int   sizeof_hdr;    /*!< MUST be 348           */  /* int sizeof_hdr;      */   /*   0 */

 char  data_type[10]; /*!< ++UNUSED++            */  /* char data_type[10];  */   /*   4 */

 char  db_name[18];   /*!< ++UNUSED++            */  /* char db_name[18];    */   /*  14 */

 int   extents;       /*!< ++UNUSED++            */  /* int extents;         */   /*  32 */

 short session_error; /*!< ++UNUSED++            */  /* short session_error; */   /*  36 */

 char  regular;       /*!< ++UNUSED++            */  /* char regular;        */   /*  38 */

 char  dim_info;      /*!< MRI slice ordering.   */  /* char hkey_un0;       */   /*  39 */
                                      /*--- was image_dimension substruct ---*/
 short dim[8];        /*!< Data array dimensions.*/  /* short dim[8];        */   /*  40 */
 float intent_p1;     /*!< 1st intent parameter. */  /* short unused8;       */   /*  56 */
                                                     /* short unused9;       */
 float intent_p2;     /*!< 2nd intent parameter. */  /* short unused10;      */   /*  60 */
                                                     /* short unused11;      */
 float intent_p3;     /*!< 3rd intent parameter. */  /* short unused12;      */   /*  64 */
                                                     /* short unused13;      */
 short intent_code;   /*!< NIFTIINTENT code.     */  /* short unused14;      */   /*  68 */
 short datatype;      /*!< Defines data type!    */  /* short datatype;      */   /*  70 */
 short bitpix;        /*!< Number bits/voxel.    */  /* short bitpix;        */   /*  72 */
 short slice_start;   /*!< First slice index.    */  /* short dim_un0;       */   /*  74 */
 float pixdim[8];     /*!< Grid spacings.        */  /* float pixdim[8];     */   /*  76 */
 float vox_offset;    /*!< Offset into .nii file */  /* float vox_offset;    */   /* 108 */
 float scl_slope;     /*!< Data scaling: slope.  */  /* float funused1;      */   /* 112 */
 float scl_inter;     /*!< Data scaling: offset. */  /* float funused2;      */   /* 116 */
 short slice_end;     /*!< Last slice index.     */  /* float funused3;      */   /* 120 */
 char  slice_code;    /*!< Slice timing order.   */                               /* 122 */
 char  xyzt_units;    /*!< Units of pixdim[1..4] */                               /* 123 */
 float cal_max;       /*!< Max display intensity */  /* float cal_max;       */   /* 124 */
 float cal_min;       /*!< Min display intensity */  /* float cal_min;       */   /* 128 */
 float slice_duration;/*!< Time for 1 slice.     */  /* float compressed;    */   /* 132 */
 float toffset;       /*!< Time axis shift.      */  /* float verified;      */   /* 136 */
 int   glmax;         /*!< ++UNUSED++            */  /* int glmax;           */   /* 140 */
 int   glmin;         /*!< ++UNUSED++            */  /* int glmin;           */   /* 144 */

                                         /*--- was data_history substruct ---*/
 char  descrip[80];   /*!< any text you like.    */  /* char descrip[80];    */   /* 148 */
 char  aux_file[24];  /*!< auxiliary filename.   */  /* char aux_file[24];   */   /* 228 */




 short qform_code;    /*!< NIFTIXFORM code.      */  /*-- all ANALYZE 7.5 ---*/   /* 252 */
 short sform_code;    /*!< NIFTIXFORM code.      */  /*   fields below here  */   /* 254 */
                                                     /*   are replaced       */
 float quatern_b;     /*!< Quaternion b param.    */                              /* 256 */
 float quatern_c;     /*!< Quaternion c param.    */                              /* 260 */
 float quatern_d;     /*!< Quaternion d param.    */                              /* 264 */
 float qoffset_x;     /*!< Quaternion x shift.    */                              /* 268 */
 float qoffset_y;     /*!< Quaternion y shift.    */                              /* 272 */
 float qoffset_z;     /*!< Quaternion z shift.    */                              /* 276 */


 float srow_x[4];     /*!< 1st row affine transform.   */                         /* 280 */
 float srow_y[4];     /*!< 2nd row affine transform.   */                         /* 296 */
 float srow_z[4];     /*!< 3rd row affine transform.   */                         /* 312 */


 char intent_name[16];/*!< name or meaning of data.  */                         /* 328 */


 char magic[4];       /*!< MUST be "ni1\0" or "n+1\0". */                         /* 344 */

 char extension[4];   

} nii_hdr ;
dim = [256 256 65 31];
res = [0.875 0.875 2.2];
res_new = [2 2 2];    
dim_new = [112 112 72];
 
[X Y Z] = meshgrid(res(1)*(0:dim(1)-1),res(2)*(0:dim(2)-1),res(3)*(0:dim(3)-1));
[XI YI ZI] = meshgrid(res_new(1)*(0:dim_new(1)-1),res_new(2)*(0:dim_new(2)-1),res_new(3)*(0:dim_new(3)-1));
 
hdr = spm_vol('DTI.nii');
dwi = spm_read_vols(hdr);
resample_dwi = zeros(dim_new);
 
for i = 1:dim(4)
    resample_dwi(:,:,:,i) = interp3(X,Y,Z,squeeze(dwi(:,:,:,i)),XI,YI,ZI,'nearest');
end

new_mat = [-2 0 0 113; 0 2 0 -113; 0 0 2 -73; 0 0 0 1];

for i = 1:dim(4)
    hdr(i).fname = 'DTI_norm.nii';
    hdr(i).dim = dim_new;
    hdr(i).mat = new_mat;
    spm_write_vol(hdr(i),resample_dwi(:,:,:,i));
end

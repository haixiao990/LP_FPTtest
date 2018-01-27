%This file is used for whole brain FACT tracking in the first section
%then, transfer the coordinates into trk file in the second section;
%Input:
%	EigenVec, FA, Dimension, resolution, threshold.
%Output:
%	whole brain trk files
%Minhui Ouyang
%Date: 04/17/14

%% First section:
tic
% image dimension and resolution
DIM =[256,256,60];
VOX =[0.875,0.875,2.2];
% read vec file
fid1 = fopen('C:\Users\Minhui\Documents\Research Project\Fiber_tracking\Data\3t2169\EigenVec_0.dat','r');
vec = fread(fid1,'float');
vec = reshape(vec,[3,DIM(1),DIM(2),DIM(3)]);
% read FA file
fid2 = fopen('C:\Users\Minhui\Documents\Research Project\Fiber_tracking\Data\3t2169\FA.dat','r');
fa = fread(fid2,'float');
fa = reshape(fa,[DIM(1),DIM(2),DIM(3)]);
% set up threshold
fa_th = 0.02;
angle_th = 0.8727;  % 50*pi/180  the angle threshold is 50
%% Fact tracking 
step = 0.1;   % stepwise
count = 0;
for numx = 1:DIM(1)
    for numy = 1:DIM(2)
        for numz = 1:DIM(3)
            if fa(numx,numy,numz) < fa_th
                continue;
            else      
        %% Tracing from both direction forward and backward
                count = count +1;            
                track(count).loc=[];    
                track(count).num=[];
                temploc=[];
                temp_voxel =[numx,numy,numz];
                  for direction = 0:1
                    % find out start index
                    XI = numx;
                    YI = numy;
                    ZI = numz;
                    % start point value
                    x0 = XI+0.5;
                    y0 = YI+0.5;
                    z0 = ZI+0.5;
                    % add step point value
                    x = x0;
                    y = y0;
                    z = z0;
                    if direction == 0
                       % track(count).loc = [track(count).loc;x,y,z];
                       temploc = [temploc;x,y,z];
                        vec_now = vec(:,XI,YI,ZI);
                        swap = 1;
                    else
                        %track(count).loc = track(count).loc;
                        temploc = temploc;
                        vec_now = -vec(:,XI,YI,ZI);
                        swap = -1;
                    end
                    while fa(XI,YI,ZI) > fa_th 
                        x = x0 + vec_now(1)*step;
                        y = y0 + vec_now(2)*step;
                        z = z0 + vec_now(3)*step;                    
                        if fix(x) == XI+1 || fix(x) == XI-1 || fix(y) == YI+1 || fix(y) == YI-1 || fix(z) == ZI+1 || fix(z) == ZI-1  % hit a boundary 
                             vec_now = vec(:,XI,YI,ZI).*swap;
                             % find index next original step                             
                            if fix(x) == XI+1                                
                                x = fix(x);
                                y = y0 + vec_now(2)*((x-x0)/vec_now(1));
                                z = z0 + vec_now(3)*((x-x0)/vec_now(1));
                                XI_next = XI+1;YI_next = fix(y);ZI_next = fix(z); 
                                if vec_now(2)*((x-x0)/vec_now(1))>1 || vec_now(3)*((x-x0)/vec_now(1))>1
                                    break;
                                end
                            elseif fix(x) == XI-1                                
                                x = XI;
                                y = y0 + vec_now(2)*((x-x0)/vec_now(1));
                                z = z0 + vec_now(3)*((x-x0)/vec_now(1));
                                XI_next = XI-1;YI_next = fix(y);ZI_next = fix(z); 
                                if vec_now(2)*((x-x0)/vec_now(1))>1 || vec_now(3)*((x-x0)/vec_now(1))>1
                                    break;
                                end
                            elseif fix(y) == YI+1 
                                y = fix(y);
                                x = x0 + vec_now(1)*((y-y0)/vec_now(2));             
                                z = z0 + vec_now(3)*((y-y0)/vec_now(2)); 
                                YI_next = YI+1; XI_next = fix(x);ZI_next = fix(z);
                                if vec_now(1)*((y-y0)/vec_now(2))>1 || vec_now(3)*((y-y0)/vec_now(2))>1
                                    break;
                                end
                            elseif fix(y) == YI-1  
                                y = YI;
                                x = x0 + vec_now(1)*((y-y0)/vec_now(2));             
                                z = z0 + vec_now(3)*((y-y0)/vec_now(2));
                                YI_next = YI-1;XI_next = fix(x);ZI_next = fix(z);
                                if vec_now(1)*((y-y0)/vec_now(2))>1 || vec_now(3)*((y-y0)/vec_now(2))>1
                                    break;
                                end                                
                            elseif fix(z) == ZI+1
                                z = fix(z);
                                x = x0 + vec_now(1)*((z-z0)/vec_now(3));
                                y = y0 + vec_now(2)*((z-z0)/vec_now(3));   
                                ZI_next = ZI+1;XI_next = fix(x);YI_next = fix(y);
                                if vec_now(1)*((z-z0)/vec_now(3))>1 || vec_now(2)*((z-z0)/vec_now(3))>1
                                    break;
                                end                                
                            else 
                                z = ZI;
                                x = x0 + vec_now(1)*((z-z0)/vec_now(3));
                                y = y0 + vec_now(2)*((z-z0)/vec_now(3)); 
                                ZI_next = ZI-1;XI_next = fix(x);YI_next = fix(y); 
                                if vec_now(1)*((z-z0)/vec_now(3))>1 || vec_now(2)*((z-z0)/vec_now(3))>1
                                    break;
                                end                                 
                            end 
                            % hit the whole brain boundary, just exit
                            if  XI_next<=0 || XI_next> DIM(1) || YI_next <=0 ||YI_next >DIM(2)|| ZI_next <=0 || ZI_next> DIM(3)
                                break;
                            end
                             % find out the angle
                            vec_next = vec(:,XI_next,YI_next,ZI_next);
                            if sum(vec_now.*vec_next) <0
                                vec_next = -vec_next;
                                swap = -1;
                            else 
                                swap = 1;
                            end
                            angle = acos(abs(sum(vec_now.*vec_next))/(sqrt(sum(vec_now.^2))*sqrt(sum(vec_next.^2))));  % dot product to find angle
                            % next voxel index
                            XI = XI_next;  
                            YI = YI_next;
                            ZI = ZI_next;
                            vec_now = vec_next;
                            % end point 
                            x0 = x;
                            y0 = y;
                            z0 = z;
                            if direction == 0
                               temploc = [temploc;x,y,z];
                               % noise cause dead loop solution 
                               if temploc(size(temploc,1),:) == temploc(size(temploc,1)-1,:) 
                                   temploc = temploc(1:size(temploc,1)-1,:);
                                   break;
                               end
                            else
                                 temploc=[x,y,z;temploc];
                               if temploc(1,:)==temploc(2,:) 
                                   temploc=temploc(2:size(temploc,1),:);
                                   break;
                               end
                            end
                            % whether the new voxel pass the angle threshold       
                            if angle > angle_th
                                break;
                            end
                            ttt(:,1)=temp_voxel(:,1)-XI;
                            ttt(:,2)=temp_voxel(:,2)-YI;
                            ttt(:,3)=temp_voxel(:,3)-ZI;
                            % find whether this voxel has existed
                            if find(sum(abs(ttt),2)==0)==1
                                ttt=[];
                                break;
                            else
                                ttt=[];
                                temp_voxel = [temp_voxel;XI YI ZI];
                            end
                        else
                            if direction == 0
                                temploc = [temploc;x,y,z];                                
                            else
                                temploc=[x,y,z;temploc];
                            end 
                            x0 = x;
                            y0 = y;
                            z0 = z;
                        end
                    end
                  end        
          %% finish forward and backward path
                 track(count).num = size(temploc,1);
                 track(count).loc = 2.*temploc;
                 [numx,numy,numz]
            end
        end
    end
end
toc

%% Second section:
% The following section is used to transfer the track.coordinates into trk file

fid = fopen('3t2169_whole_brain_FACT.trk', 'wb');
% write the header of .trk file
fwrite(fid, ['T';'R';'A';'C';'K';char(0)], 'char'); % id_string[6],6
fwrite(fid, [DIM(1); DIM(2); DIM(3)], 'int16');     % dim[3], short int, 6
fwrite(fid, [VOX(1); VOX(2); VOX(3)], 'float');     % voxel_size[3], float, 12, Voxel size of the image volume.
fwrite(fid, [0; 0; 0], 'float');                    % origin[3], float, 12, Origin of the image volume. This field is not yet being used by TrackVis.
fwrite(fid, 0, 'int16');                            % n_scalars, short int, 2, Number of scalars saved at each track point (besides x, y and z coordinates).
fwrite(fid, empty_char(200,1), 'char');             % scalar_name[10][20], char, 200, Name of each scalar.
fwrite(fid, 0, 'int16');                            % n_properties, short int, 2, Number of properties saved at each track.
fwrite(fid, empty_char(200,1), 'char');             % property_name[10][20], char, 200, Name of each property
fwrite(fid, [VOX(1) 0 0 0 0 VOX(2) 0 0 0 0 VOX(3) 0 0 0 0 1], 'float'); % vox_to_ras[4][4], float, 64, 4x4 matrix for voxel to RAS (crs to xyz) transformation. If vox_to_ras[3][3] is 0, it means the matrix is not recorded. This field is added from
fwrite(fid, empty_char(444,1), 'char');             % reserved[444], char, 444, Reserved space for future version.
fwrite(fid, ['L';'P';'S';char(0)], 'char');         % voxel_order[4], char, 4, Storing order of the original image data. Explained here.
fwrite(fid, empty_char(4,1), 'char');               % pad2[4], char, 4, Paddings.           
fwrite(fid, [1;0;0;0;1;0], 'float');                % image_orientation_patient[6], float, 24, Image orientation of the original image. As defined in the DICOM header.
fwrite(fid, empty_char(2,1), 'char');               % pad1[2], char, 2, Paddings.
fwrite(fid, 0, 'uchar');                            % invert_x, unsigned char, 1,
fwrite(fid, 0, 'uchar');                            % invert_y, unsigned char, 1
fwrite(fid, 0, 'uchar');                            % invert_x, unsigned char, 1, As above.
fwrite(fid, 0, 'uchar');                            % swap_xy, unsigned char, 1, As above.
fwrite(fid, 0, 'uchar');                            % swap_yz, unsigned char, 1, As above.
fwrite(fid, 0, 'uchar');                            % swap_zx, unsigned char, 1, As above.
fwrite(fid, 0, 'int');                              %fwrite(fid, length(fibers), 'int');% n_count, int, 4, Number of tracks stored in this track file. 0 means the number was NOT stored.
fwrite(fid, 2, 'int');                              % version, int, 4, Version number. Current version is 2.
fwrite(fid, 1000, 'int');                           % hdr_size, int, 4, Size of the header. Used to determine byte swap. Should be 1000.
% write the data part of .trk file
 for i=1:count
     fwrite(fid, track(1,i).num, 'int'); 
     for j=1:track(1,i).num
         fwrite(fid,track(1,i).loc(j,:),'float');
     end
 end
 fclose(fid);
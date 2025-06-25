from .Globals import *

if G.GPU_MODE:
    cp.cuda.Device(G.DEVICE).use()

class Landmarks(object):
    mode_return_type = lambda x, y: x.astype(getattr(cp, y)) if G.GPU_MODE else lambda x, y: x.astype(getattr(np, y))
    mode_create_array = lambda x: cp.array(x) if G.GPU_MODE else lambda x: np.array(x)
    mode_function = lambda x: getattr(cp, x.__name__) if G.GPU_MODE else getattr(np, x.__name__)
    mode_value = lambda x: getattr(cp, x.__str__()) if G.GPU_MODE else getattr(np, x.__str__())
    cp_to_np = lambda x: cp.asnumpy(x) if G.GPU_MODE else lambda x: x
    np_to_cp = lambda x: cp.asarray(x) if G.GPU_MODE else lambda x: x

    def __init__(self,
                 volume_name:str,
                 physical_center:np.ndarray,
                 physical_start:np.ndarray,
                 physical_steps:np.ndarray,
                 texture_center:int|float = 500,
                 group_name: str = 'AllVolumes',
                 volume_file: Path = None,
                 max_landmarks:int = 2**16):

        self.max_landmarks = max_landmarks
        self.texture_center = texture_center
        self.volume_name = volume_name
        self.group_name = group_name
        self.file_path = volume_file.parent.joinpath(f'{group_name}-{self.volume_name}-Landmarks.csv')
        self.physical_center = physical_center
        self.physical_start = physical_start
        self.physical_steps = physical_steps
        self.landmarks_table = f'{volume_name}_landmarks_table'

        self.landmark_image_coords = np.zeros((self.max_landmarks, 4), dtype = np.float32) #(x, y, z, hu), obtained using VolumeLayerGroups.get_drawing_pos_coords
        self.landmark_physical_coords = np.zeros((self.max_landmarks, 4), dtype = np.float32) # Position in physical mm space, get_physical_pos_coords
        self.landmark_voxel_coords = np.zeros((self.max_landmarks, 3), dtype = np.float32) # Actual position in image voxel index space, get_physical_voxel_coords
        self.landmark_drawing_coords = np.zeros((self.max_landmarks, 3), dtype = np.float32) #(x, y, unscaled distance)
        self.landmark_distances = np.zeros(self.max_landmarks, dtype = np.float32) # Scaled distances
        self.landmark_quaternions = np.zeros((self.max_landmarks, 4), dtype = np.float32)
        self.landmark_norms = np.zeros((self.max_landmarks, 3), dtype = np.float32)
        self.landmark_geometries = np.zeros((self.max_landmarks, 3), dtype = np.float32)
        self.landmark_sizes = np.zeros(self.max_landmarks, dtype = np.float32)
        self.landmark_rgba = np.zeros((self.max_landmarks, 4), dtype = np.float32) #(r, g, b, a)
        self.landmark_patches = {}
        self.landmark_show = np.zeros(self.max_landmarks, dtype = np.int8) #(true, false)
        self.landmark_dict = {} # {volume_name||array_index: array_index}
        self.landmark_last_tag = ''
        self.landmark_valid_indices = []

        self.landmark_index = int(0)
        self.number_of_landmarks = int(0)
        self.number_of_landmarks_visible = int(0)

    def add_landmark(self, 
                     drawing_coords: tuple[int],
                     image_coords: np.ndarray, 
                     image_coords_hu: float, 
                     voxel_coords: np.ndarray,
                     quaternion,
                     viewplane_norm: np.ndarray, 
                     draw_layer: str,
                     color = dpg.get_value('landmark_color_picker'),
                     size:float = 5.0, 
                     geometry: np.ndarray = np.ones(3, dtype = np.float32),
                     landmark_patch: np.ndarray = np.zeros((11, 11), dtype = np.float32), 
                     patch_size: int = 11,
                     show_landmark = True):
        """
        Parameters: 
        ----------
            drawing_coords: tuple[int, int]
                Position on the drawing, eg mouse position. 

            image_coords: ndarray[float, float, float]
                Position on the view_plane. Can be obtained using get_drawing_pos_coords()

            image_coords_hu: float
                HU value at image_coords. 

            voxel_coords: ndarray[float, float, float]
                Voxel coordinates in physical units (eg, millimeters)

            quaternion: QuaternionicArray
                Quaternion applied when selecting the landmark

            viewplane_norm: ndarray[float, float, float]
                Norm of the viewplane when the landmark was added. 

            draw_layer: str
                Tag of the drawlayer to add the landmark to

            color: ndarray[float, float, float, float]
                Initial RGBA color of the landmark
                Default is dpg.get_value('landmark_color_picker')

            size: float
                Radius of the landmark in pixels
                Default is 5.0

            preview: ndarray[N, N] or None
                NxN image of the landmark when added. 

            show_landmark: bool
                Show or hide the landmark
                Default is True. 
        """

        self.landmark_image_coords[self.landmark_index, :3] = image_coords[:]
        self.landmark_image_coords[self.landmark_index, 3] = image_coords_hu
        self.landmark_physical_coords[self.landmark_index, :3] = image_coords[:]
        self.landmark_physical_coords[self.landmark_index, 3] = image_coords_hu
        self.landmark_voxel_coords[self.landmark_index] = voxel_coords[:]
        self.landmark_drawing_coords[self.landmark_index][0] = float(drawing_coords[1])
        self.landmark_drawing_coords[self.landmark_index][1] = float(drawing_coords[0])
        self.landmark_drawing_coords[self.landmark_index][2] = 0.0
        self.landmark_distances[self.landmark_index] = 0.0
        
        self.landmark_quaternions[self.landmark_index] = np.array(quaternion)[:]
        self.landmark_norms[self.landmark_index] = Landmarks.cp_to_np(viewplane_norm)[:]
        self.landmark_sizes[self.landmark_index] = 1.0*size
        self.landmark_geometries[self.landmark_index] = 1.0*geometry
        self.landmark_rgba[self.landmark_index] = np.array(color)[:]
        self.landmark_show[self.landmark_index] = int(show_landmark)

        # print('Landmarks Message: Adding Landmark:')
        # print(f'\tLandmark {self.landmark_index} Color: {self.landmark_rgba[self.landmark_index]}')
        self.landmark_last_tag = f'{self.volume_name}||{self.landmark_index}'
        

        landmark_circle = self.draw_landmark(drawing_coords,
                                             size * np.mean(geometry),
                                             draw_layer,
                                             self.landmark_last_tag,
                                             color)

        self.landmark_dict[landmark_circle] = self.landmark_index
        patch_texture_tag = f'{landmark_circle}||PatchTexture'
        self.landmark_patches[patch_texture_tag] = [patch_size, 1.0*landmark_patch]

        self.landmark_valid_indices.append(int(1 * self.landmark_index))
        self.landmark_index += 1        
        self.number_of_landmarks += 1
        self.number_of_landmarks_visible = np.sum(self.landmark_show)


    def draw_landmark(self, 
                      drawing_coords: tuple[int|float],
                      radius: int|float,
                      draw_layer: str,
                      tag: str|int,
                      color: tuple[float]) -> str|int:
        
        landmark_circle = dpg.draw_circle(
            drawing_coords,
            radius = radius, 
            color = color, 
            parent = draw_layer,
            tag = tag)
        
        return landmark_circle


    def xform_voxel_to_origin_coords(self, 
                                     landmark_voxel_coords: np.ndarray):
        
        return landmark_voxel_coords[:, :3] 


    def xform_voxel_to_image_coords(self, 
                                    landmark_voxel_coords: np.ndarray):
        
        return (landmark_voxel_coords[:, :3] * self.physical_steps) + self.physical_start - self.physical_center
    

    def xform_image_to_voxel_coords(self, 
                                     landmark_image_coords: np.ndarray):
        
        return ((landmark_image_coords[:, :3] + self.physical_center) - self.physical_start) / self.physical_steps


    def get_landmark_drawing_coords(self, 
                                    landmark_image_coords: np.ndarray, # x, y, z, hu
                                    origin_vector: np.ndarray, 
                                    quaternion, 
                                    geometry_vector: np.ndarray) -> np.ndarray:
        
        """
        We store the landmarks using their voxel coordinates, not the image coordinates. 
        We need to find the drawing coords using the voxel coordinates, which are initially obtained from the 
        drawing position using VolumeLayer.get_physical_voxel_coords. 

        The math is: 

            physical_coords = image_pos_coords + ctvolume.physical_center
            voxel_coords    = (physical_coords - ctvolume.physical_start) / ctvolume.pixel_steps
                            = ((image_pos_coords + ctvolume.physical_center) - ctvolume.physical_start) / ctvolume.pixel_steps

        So we need to extract the image_pos_coords:
            image_pos_coords = (voxel_coords * ctvolume.pixel_steps) + ctvolume.physical_start - ctvolume.physical_center

        """

        shift = quaternion.inverse.rotate(landmark_image_coords[:, :3] - origin_vector)
        shift[:, :2] *= geometry_vector[:2]
        landmark_start_coords = np.array([self.texture_center, self.texture_center, 0.0])
        landmark_drawing_coords = landmark_start_coords + shift
        
        return landmark_drawing_coords


    def update_landmarks(self, 
                         origin_vector: np.ndarray,
                         quaternion,
                         geometry_vector: np.ndarray):
        """
        Updates the drawing position, alpha, size, and shape of all landmarks. 

        Parameters:
        ----------
            origin_vector: ndarray[float, float, float]
                Current origin position. 
                VolumeLayer.get_crosshair_coords is most reliable method. 

            quaternion: QuaternionicArray
                Current quaternion.  Applies rotation to landmarks. 

            geometry_vector: ndarray[float, float, float]
                Current geometry vector for scaling positions and rotations. 

        """

        shift = quaternion.inverse.rotate(self.landmark_image_coords[:self.landmark_index, :3] - origin_vector)
        shift[:self.landmark_index, :2] *= geometry_vector[:2]
        landmark_start_coords = np.array([self.texture_center, self.texture_center, 0.0])
        landmark_color_picker = 1.0*np.array(dpg.get_value('landmark_color_picker'))

        self.landmark_drawing_coords[:self.landmark_index] = landmark_start_coords + shift
        self.landmark_distances[:self.landmark_index] = np.abs(self.landmark_drawing_coords[:self.landmark_index, 2]) / geometry_vector[2]
        self.landmark_rgba[:self.landmark_index,3] = self.calculate_landmark_fade(
                                                            landmark_color_picker[3], #self.landmark_rgba[:,3], 
                                                            self.landmark_distances[:self.landmark_index])

        # print('Landmarks Message: update_landmarks')

        for landmark_id, landmark_index in self.landmark_dict.items():
            dpg.configure_item(landmark_id, 
                               radius = self.landmark_sizes[landmark_index] * np.mean(geometry_vector),
                               center = (self.landmark_drawing_coords[landmark_index][1], 
                                         self.landmark_drawing_coords[landmark_index][0]),
                               color = self.landmark_rgba[landmark_index].tolist())


    def update_landmark_colors(self, 
                               landmark_id: int|None = None):
        
        landmark_color_picker = 1.0*np.array(dpg.get_value('landmark_color_picker'))

        if isinstance(landmark_id, int):
            landmark_index = self.landmark_dict[landmark_id]
            self.landmark_rgba[landmark_index, :3] = landmark_color_picker[:3]
            self.landmark_rgba[landmark_index, :3] = self.calculate_landmark_fade(landmark_color_picker[3], 
                                                                                  self.landmark_distances[landmark_index])

            dpg.configure_item(landmark_id,
                               color = self.landmark_rgba[landmark_index].tolist())
            return
        
        self.landmark_rgba[:self.landmark_index, :3] = landmark_color_picker[:3]
        self.landmark_rgba[:self.landmark_index, 3]  = self.calculate_landmark_fade(landmark_color_picker[3], 
                                                                                    self.landmark_distances[:self.landmark_index])

        for landmark_id, landmark_index in self.landmark_dict.items():
            if self.landmark_show[landmark_index]:
                dpg.configure_item(landmark_id,
                                color = self.landmark_rgba[landmark_index].tolist())

    def calculate_landmark_fade(self, 
                                landmark_alpha,
                                landmark_distance):
        return landmark_alpha * np.exp(-(2.0 / self.landmark_sizes[:self.landmark_index]) * landmark_distance / np.round(dpg.get_value('landmark_opacity_factor_input'), decimals = 1) + 0.0)

    def set_landmark_opacity(self, view_plane, override = None, override_value = 0):

        if type(override) != type(None):
            self.landmark_rgba[:,3] *= override_value

    def get_last_landmark(self) -> list:
        if self.landmark_index > 0:

            landmark_info_list = [self.volume_name, 
                                  self.landmark_index - 1, 
                                  self.landmark_image_coords[self.landmark_index - 1, :].round(3), 
                                  self.landmark_last_tag,
                                  self.get_patch_tag(self.landmark_last_tag),
                                  self.landmark_patches[self.get_patch_tag(self.landmark_last_tag)]]

            # print('Landmark Message: get_last_landmark')
            # print(f'\tVolume Name       : {landmark_info_list[0]}')
            # print(f'\tLandmark Index    : {landmark_info_list[1]}')
            # print(f'\tLandmark Coords   : {landmark_info_list[2]}')
            # print(f'\tLandmark ID       : {landmark_info_list[3]}')
            # print(f'\tLandmark Patch ID : {landmark_info_list[4]}')
            # print(f'\tLandmark Patch    : {landmark_info_list[5]}')

            return landmark_info_list
        

    def get_patch_tag(self, 
                      landmark_tag):
        
        return f'{landmark_tag}||PatchTexture'


    def get_landmarks_info(self) -> dict:
        """
        Returns landmarks_info_dict
        """
        landmarks_info_dict = {'x': self.landmark_image_coords.round(3)[self.landmark_valid_indices, 1],
                               'y': -1.0*self.landmark_image_coords.round(3)[self.landmark_valid_indices, 0] + 0.0, 
                               'z': self.landmark_image_coords.round(3)[self.landmark_valid_indices, 2],
                               'hu': self.landmark_image_coords.round(3)[self.landmark_valid_indices, 3],
                               'norm_x': self.landmark_norms.round(3)[self.landmark_valid_indices, 0],
                               'norm_y': self.landmark_norms.round(3)[self.landmark_valid_indices, 1],
                               'norm_z': self.landmark_norms.round(3)[self.landmark_valid_indices, 2],
                               'qtn_a': self.landmark_quaternions.round(3)[self.landmark_valid_indices, 0],
                               'qtn_b': self.landmark_quaternions.round(3)[self.landmark_valid_indices, 1],
                               'qtn_c': self.landmark_quaternions.round(3)[self.landmark_valid_indices, 2],
                               'qtn_d': self.landmark_quaternions.round(3)[self.landmark_valid_indices, 3],
                               'pix_x': self.landmark_geometries.round(3)[self.landmark_valid_indices, 1],
                               'pix_y': self.landmark_geometries.round(3)[self.landmark_valid_indices, 0],
                               'pix_z': self.landmark_geometries.round(3)[self.landmark_valid_indices, 2]}
        
        return landmarks_info_dict
    
    def get_landmark_preview(self, 
                             landmark_index:int, 
                             view_width:float = 5.0, 
                             view_quaternion: qtn.QuaternionicArray = None) -> np.ndarray:

        pass

    def load_landmark_data(self, 
                           current_origin,
                           current_quaternion,
                           current_geometry,
                           file_path: Path) -> dict:
        
        data_dict = self.read_landmark_csv(file_path)

        if data_dict['n_landmarks'] == 0:
            print('Landmarks Message: No Landmarks to Load')
            return False

        # data should be of shape (n_landmarks, 14)
        data = data_dict['data'] #vx, vy, vz, hu, nz, ny, nz, qa, qb, qc, qd, px, py, pz

        loaded_data = {'image_coords': np.zeros((len(data), 4), dtype = np.float32),
                       'voxel_coords': np.zeros((len(data), 3), dtype = np.float32),
                       'drawing_coords': np.zeros((len(data), 3), dtype = np.float32),
                       'norms': np.zeros((len(data), 3), dtype = np.float32),
                       'quaternions': np.zeros((len(data), 4), dtype = np.float32),
                       'geometries': np.zeros((len(data), 3), dtype = np.float32)}

        loaded_data['voxel_coords'] = data[:, :4]
        loaded_data['norms'] = data[:, 4:7]
        loaded_data['quaternions'] = data[:, 7:11]
        loaded_data['geometries'] = data[:, 11:]

        loaded_data['image_coords'][:,1] = 1.0 * loaded_data['voxel_coords'][:,0] + 0.0
        loaded_data['image_coords'][:,0] = -1.0 * loaded_data['voxel_coords'][:,1] + 0.0
        loaded_data['image_coords'][:,2] = 1.0 * loaded_data['voxel_coords'][:,2] + 0.0
        loaded_data['image_coords'][:,3] = 1.0 * loaded_data['voxel_coords'][:,3] + 0.0
        # loaded_data['image_coords'][:,:3] = self.xform_voxel_to_image_coords(loaded_data['voxel_coords'])
        # loaded_data['image_coords'][:,1] *= -1.0 + 0.0
        loaded_data['drawing_coords'][:] = self.get_landmark_drawing_coords(loaded_data['image_coords'], 
                                                                            current_origin,
                                                                            current_quaternion, 
                                                                            current_geometry)

        return loaded_data
        

    def format_landmark_header_line(self, 
                                    header_line: str):
        
        return f'{header_line}'.replace(' ', '').strip('[').strip(']')
    

    def format_landmark_data_line(self, 
                                  landmark_data:np.ndarray,
                                  newline: bool = False):
         

         if newline:
             return f"{landmark_data.tolist()}\n".replace(' ', '').strip('[').strip(']')
         
         return f"{landmark_data.tolist()}".replace(' ', '').strip('[').strip(']')


    def update_landmark_csv(self, 
                            landmark_file:Path, 
                            landmark_data:np.ndarray):
        with open(landmark_file, mode = 'a') as file:
            print(self.format_landmark_data_line(landmark_data, newline = True), file = file, flush = True)


    def save_landmark_csv(self, 
                          landmark_file: Path, 
                          vol_name: str,
                          affine: np.ndarray,
                          data_header: str,
                          data: np.ndarray,
                          mode = 'w'):
        
        n_landmarks = len(data)

        lines = f'#{vol_name}\n#{affine.tolist()}\n#{n_landmarks}\n{self.format_landmark_header_line(data_header)}\n'

        for landmark_data in data:
            lines = f"{lines}{self.format_landmark_data_line(landmark_data)}\n"

        with open(landmark_file, mode = mode) as file:
            print(lines, file = file, flush = True, end = '')


    def read_landmark_csv(self, 
                        landmark_file: Path, 
                        comment_lines: int = 2, 
                        header_line: bool = True):

        return_dict = {'vol_name': '',
                       'affine': np.eye(4, 4),
                       'n_landmarks': 0,
                       'data_header': [],
                       'data': []}
        
        with open(landmark_file) as file:
        
            return_dict['vol_name'] = file.readline().split('#')[1][:-1]
            affine = file.readline().split('#')[1][:-1]
            return_dict['n_landmarks'] = int(file.readline().split('#')[1][:-1])
            stripped_affine = affine.replace(' ', '').lstrip('[').rstrip(']').replace('],[', ' ').split(' ')
            
            for index, s_affine in enumerate(stripped_affine):
                return_dict['affine'][index] = np.fromstring(s_affine, sep = ',')
            
            return_dict['data_header'] = file.readline()[:-1].split(',')
            for data_line in file:
                return_dict['data'].append(np.fromstring(data_line[:-1], sep = ','))
        
        return_dict['data'] = np.array(return_dict['data'])

        return return_dict
    

    def save_landmarks(self, 
                       file_path:Path,
                       volume_name: str,
                       volume_affine: np.ndarray):
        
        if file_path.suffix not in ['.csv', '.txt']:
            print(f'Landmarks Message: save_landmarks')
            print(f'\tFile extension {file_path.suffix} not allowed!')

            return
        
        landmarks_info_dict = self.get_landmarks_info()

        data_header = list(landmarks_info_dict.keys())
        landmark_data = np.array(list(landmarks_info_dict.values())).T
        
        self.save_landmark_csv(file_path, 
                               volume_name,
                               volume_affine,
                               data_header,
                               landmark_data, 
                               mode = 'w')

        print('Landmarks Message: Landmarks saved at: ')
        print(f'\t{file_path}')


    def hide_landmarks(self):
        self.landmark_show[:self.landmark_index] = 0
        for landmark_id in self.landmark_dict.keys():
            dpg.hide_item(landmark_id)

    def show_landmarks(self):
        self.landmark_show[:self.landmark_index] = 1
        for landmark_id in self.landmark_dict.keys():
            dpg.show_item(landmark_id)

    def delete_landmark(self, landmark_id):

        print(f'Landmark Message: Deleting Landmark {landmark_id}')

        landmark_index = self.landmark_dict[landmark_id]
        self.landmark_image_coords[landmark_index] *= 0.0
        self.landmark_voxel_coords[landmark_index] *= 0.0
        self.landmark_physical_coords[landmark_index] *= 0.0
        self.landmark_drawing_coords[landmark_index] *= 0.0
        self.landmark_distances[landmark_index] *= 0.0
        self.landmark_quaternions[landmark_index] *= 0.0
        self.landmark_norms[self.landmark_index] *= 0.0
        self.landmark_sizes[landmark_index] *= 0.0
        self.landmark_rgba[landmark_index] *= 0.0
        self.landmark_show[landmark_index] *= 0
        self.landmark_geometries[self.landmark_index] *= 0.0

        if dpg.does_item_exist(landmark_id):
            dpg.delete_item(landmark_id)
        if dpg.does_alias_exist(landmark_id):
            dpg.remove_alias(landmark_id)

        del self.landmark_dict[landmark_id]

        self.landmark_valid_indices.remove(int(landmark_index))
        self.number_of_landmarks -= 1


    def delete_all_landmarks(self):
        for landmark_id in list(self.landmark_dict.keys()):
            self.delete_landmark(landmark_id)

        print(f'Landmarks Message: Deleted landmarks for {self.volume_name}')
        print(f'\tNumber of Landmarks: {self.number_of_landmarks}')

    def _cleanup_(self):
        self.delete_all_landmarks()
        dict_keys = list(self.__dict__.keys())
        while len(dict_keys) > 0:
            attrib_key = dict_keys.pop()
            setattr(self, attrib_key, None)
            delattr(self, attrib_key)
            cp._default_memory_pool.free_all_blocks()
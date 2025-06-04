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
                 texture_center:int|float = 500,
                 max_landmarks:int = 2**16):

        self.max_landmarks = max_landmarks
        self.texture_center = texture_center
        self.volume_name = volume_name
        self.landmarks_table = f'{volume_name}_landmarks_table'

        self.landmark_image_coords = np.zeros((self.max_landmarks, 4), dtype = np.float32) #(x, y, z, hu)
        self.landmark_physical_coords = np.zeros((self.max_landmarks, 4), dtype = np.float32)
        self.landmark_voxel_coords = np.zeros((self.max_landmarks, 3), dtype = np.float32)
        self.landmark_drawing_coords = np.zeros((self.max_landmarks, 3), dtype = np.float32) #(x, y, unscaled distance)
        self.landmark_distances = np.zeros(self.max_landmarks, dtype = np.float32) # Scaled distances
        self.landmark_quaternions = np.zeros((self.max_landmarks, 4), dtype = np.float32)
        self.landmark_norms = np.zeros((self.max_landmarks, 3), dtype = np.float32)
        self.landmark_sizes = np.zeros(self.max_landmarks, dtype = np.float32)
        self.landmark_rgba = np.zeros((self.max_landmarks, 4), dtype = np.float32) #(r, g, b, a)
        self.landmark_patches = {}
        self.landmark_show = np.zeros(self.max_landmarks, dtype = np.int8) #(true, false)
        self.landmark_dict = {} # {volume_name||array_index: array_index}
        self.landmark_last_tag = ''

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
                     mean_geometry: float = 1.0, 
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
        self.landmark_rgba[self.landmark_index] = np.array(color)[:]
        self.landmark_show[self.landmark_index] = int(show_landmark)

        # print('Landmarks Message: Adding Landmark:')
        # print(f'\tLandmark {self.landmark_index} Color: {self.landmark_rgba[self.landmark_index]}')
        self.landmark_last_tag = f'{self.volume_name}||{self.landmark_index}'

        landmark_circle = dpg.draw_circle(
            drawing_coords,
            radius = size * mean_geometry, 
            color = color, 
            parent = draw_layer,
            tag = self.landmark_last_tag
        )

        self.landmark_dict[landmark_circle] = self.landmark_index
        patch_texture_tag = f'{landmark_circle}||PatchTexture'
        self.landmark_patches[patch_texture_tag] = [patch_size, 1.0*landmark_patch]

        self.landmark_index += 1
        self.number_of_landmarks += 1
        self.number_of_landmarks_visible = np.sum(self.landmark_show)

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
        print(f'\t{self.landmark_image_coords[:self.landmark_index, :3] = }')
        print(f'\t{self.landmark_drawing_coords[:self.landmark_index] = }')

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
                    self.landmark_image_coords[self.landmark_index - 1].round(3), 
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


    def get_landmarks_info(self) -> list:
        """
        Returns landmarks_info_dict_list
        """
        landmarks_info_dict_list = []
        for landmark_id, landmark_index in self.landmark_dict.items():
            landmarks_info_dict_list.append({'x': self.landmark_voxel_coords.round(3)[landmark_index, 1],
                                             'y': self.landmark_voxel_coords.round(3)[landmark_index, 0],
                                             'z': self.landmark_voxel_coords.round(3)[landmark_index, 2],
                                             'hu': self.landmark_image_coords.round(3)[landmark_index, 3],
                                             'norm_x':self.landmark_norms.round(3)[landmark_index, 0],
                                             'norm_y':self.landmark_norms.round(3)[landmark_index, 0],
                                             'norm_z':self.landmark_norms.round(3)[landmark_index, 0],
                                             'qtn_a': self.landmark_quaternions.round(3)[landmark_index, 0],
                                             'qtn_b': self.landmark_quaternions.round(3)[landmark_index, 1],
                                             'qtn_c': self.landmark_quaternions.round(3)[landmark_index, 2],
                                             'qtn_d': self.landmark_quaternions.round(3)[landmark_index, 3]})
        
        return landmarks_info_dict_list
    
    def get_landmark_preview(self, 
                             landmark_index:int, 
                             view_width:float = 5.0, 
                             view_quaternion: qtn.QuaternionicArray = None) -> np.ndarray:

        pass        

    def save_landmarks(self, 
                       file_path:Path,
                       sheet_name:str,
                       extension:str = 'xlsx'):
        
        if extension not in ['csv', 'txt', 'xlsx']:
            print(f'Landmarks Message: save_landmarks')
            print(f'\tFile extension {extension} not recognized!')

            return
        
        file_path:Path = file_path.with_suffix(f'.{extension}')
        landmarks_info_list = self.get_landmarks_info()
        dataframe = pandas.DataFrame(landmarks_info_list)

        if extension == 'xlsx':
            excel_writer = pandas.ExcelWriter(file_path, 
                                              mode = 'w',
                                              if_sheet_exists='replace',
                                              engine = 'xlsxwriter',
                                              engine_kwargs = {'options': {'string_to_numbers': True}})
            
            dataframe.to_excel(excel_writer, 
                               sheet_name = sheet_name,
                               index = False)

            excel_writer.close()

        else:
            dataframe.to_csv(file_path, 
                             sep = ',', 
                             index=False)

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
        self.landmark_drawing_coords[landmark_index] *= 0.0
        self.landmark_distances[landmark_index] *= 0.0
        self.landmark_quaternions[landmark_index] *= 0.0
        self.landmark_sizes[landmark_index] *= 0.0
        self.landmark_rgba[landmark_index] *= 0.0
        self.landmark_show[landmark_index] *= 0

        if dpg.does_item_exist(landmark_id):
            dpg.delete_item(landmark_id)
        if dpg.does_alias_exist(landmark_id):
            dpg.remove_alias(landmark_id)

        del self.landmark_dict[landmark_id]

        self.number_of_landmarks -= 1

    def _cleanup_(self):
        dict_keys = list(self.__dict__.keys())
        while len(dict_keys) > 0:
            attrib_key = dict_keys.pop()
            setattr(self, attrib_key, None)
            delattr(self, attrib_key)
            cp._default_memory_pool.free_all_blocks()
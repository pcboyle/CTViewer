from .Globals import *

if G.GPU_MODE:
    cp.cuda.Device(G.DEVICE).use()

class DummyValue(object):
    def __init__(self):
        pass
    
mode_return_type = lambda x, y: x.astype(getattr(cp, y)) if G.GPU_MODE else lambda x, y: x.astype(getattr(np, y))
mode_create_array = lambda x: cp.array(x) if G.GPU_MODE else lambda x: np.array(x)
mode_function = lambda x: getattr(cp, x.__name__) if G.GPU_MODE else getattr(np, x.__name__)
mode_value = lambda x: getattr(cp, x.__str__()) if G.GPU_MODE else getattr(np, x.__str__())
cp_to_np = lambda x: cp.asnumpy(x) if G.GPU_MODE else lambda x: x
np_to_cp = lambda x: cp.asarray(x) if G.GPU_MODE else lambda x: x


class IntensityInfo(object):
    def __init__(self, 
                 tag = 'Intensity',
                 default_colormap_info = {'min_intensity': {'tag': G.OPTION_TAG_DICT['min_intensity'],
                                                            'default_value': 0.0, 
                                                            'default_limits': [-1e6, 1e6 - 1]},
                                          'max_intensity': {'tag': G.OPTION_TAG_DICT['max_intensity'],
                                                            'default_value': 1200.0, 
                                                            'default_limits': [-1e6 + 1, 1e6]},
                                          'colormap': G.COLORMAP_DICT['Fire'],
                                          'colormap_reversed': False,
                                          'colormap_name': 'Fire',
                                          'colormap_scale_type': 'Linear',
                                          'colormap_scale_tag': 'colormap_scale_combo'}):
        
        self.tag: StringValue = StringValue(default_string=tag)
        self.min_intensity: OptionValue = OptionValue(**default_colormap_info['min_intensity'])
        self.max_intensity: OptionValue = OptionValue(**default_colormap_info['max_intensity'])
        # self.colormap: list[interp1d] = default_colormap_info['colormap']
        self.colormap: ColormapValue = ColormapValue(default_colormap = default_colormap_info['colormap'])
        self.colormap_name: StringValue = StringValue(default_string = default_colormap_info['colormap_name']) # eg, Fire
        self.colormap_reversed: bool = default_colormap_info['colormap_reversed']
        self.colormap_log: bool = False
        self.colormap_scale_type: StringValue = StringValue(default_string = default_colormap_info['colormap_scale_type'])
        self.colormap_scale_tag: StringValue = StringValue(default_string = default_colormap_info['colormap_scale_tag'])
        self.colormap_rescaled: bool = False
        self.window_size = 1

        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()


    def get_info(self) -> dict:

        return {'tag': self.tag,
                'min_intensity': self.min_intensity,
                'max_intensity': self.max_intensity,
                'colormap': self.colormap,
                'colormap_name':self.colormap_name,
                'colormap_reversed': self.colormap_reversed,
                'colormap_log': self.colormap_log,
                'colormap_scale_type': self.colormap_scale_type,
                'colormap_scale_tag': self.colormap_scale_tag,
                'colormap_rescaled': self.colormap_rescaled,
                'window_size': self.window_size}


    def update_window(self, 
                      min_intensity: float, 
                      max_intensity: float, 
                      window_size: float|int):
        
        self.min_intensity.update_values(min_intensity)
        self.max_intensity.update_values(max_intensity)

        if self.max_intensity.current_value - self.min_intensity.current_value < window_size:
            if self.max_intensity.difference_value == 0: # This means we are changing max_intensity.
                self.max_intensity.update_values(self.min_intensity.current_value + window_size)
            elif self.min_intensity.difference_value == 0: # This means we are changing min_intensity. 
                self.min_intensity.update_values(self.max_intensity.current_value - window_size)
        
            dpg.set_value(G.OPTION_TAG_DICT['min_intensity'], self.min_intensity.current_value)
            dpg.set_value(G.OPTION_TAG_DICT['max_intensity'], self.max_intensity.current_value)
        

    def update_colormap(self, 
                        colormap_name: str = None,
                        colormap_reversed: bool = None,
                        colormap_log: bool = None, 
                        colormap_scale_type: str = None,
                        colormap_scale_tag: str = None,
                        colormap_rescaled: bool = None
                        ):
        
        if colormap_name == None:
            colormap_name = dpg.get_value('colormap_combo')
        
        if colormap_reversed == None:
            colormap_reversed = colormap_reversed

        if colormap_scale_type == None:
            pass

        if colormap_scale_tag == None:
            pass

        if colormap_rescaled == None:
            pass

        self.colormap_name.update_values(colormap_name)
        self.colormap_scale_tag.update_values(colormap_scale_tag)
        self.colormap_scale_type.update_values(colormap_scale_type)
        self.colormap_rescaled = colormap_rescaled
        self.colormap_reversed = colormap_reversed
        self.colormap_log = colormap_log

        if self.colormap_reversed: 
            setattr(self, 'colormap', G.COLORMAP_DICT[colormap_name]['colormap_r'])
        else:
            setattr(self, 'colormap', G.COLORMAP_DICT[colormap_name]['colormap'])

        dpg.configure_item(self.colormap_scale_tag, colormap = f'colormap_{self.get_colormap_string()}')
    
    
    def copy_intensity(self, intensity_info: "IntensityInfo"):
        # print('IntensityInfo Message: Before Copying')
        # self.print()
        for value_type in ['current_value', 'previous_value', 'difference_value', 'default_value']:
            setattr(self.min_intensity, value_type, 1.0*getattr(intensity_info.min_intensity, value_type))
            setattr(self.max_intensity, value_type, 1.0*getattr(intensity_info.max_intensity, value_type))
            setattr(self.colormap_name, value_type, f'{getattr(intensity_info.colormap_name, value_type)}')
            setattr(self.colormap_scale_tag, value_type, f'{getattr(intensity_info.colormap_scale_tag, value_type)}')

        self.colormap = intensity_info.colormap
        self.colormap_reversed = intensity_info.colormap_reversed
        self.window_size = 1.0*intensity_info.window_size
        
        # print('IntensityInfo Message: After Copying')
        # self.print()


    def get_colormap_string(self):
        cmap_name = f'{G.COLORMAP_DICT[self.colormap_name.current_value]["name"]}'

        if self.colormap_reversed:
            cmap_name = f'{cmap_name}_r'
            # if self.colormap_log:
            #     cmap_name = f'{cmap_name}_log'
            # return cmap_name
        
        # if self.colormap_log:
        #     cmap_name = f'{cmap_name}_log'

        return cmap_name
    

    def set_colormap_scale_tag(self, colormap_scale_tag: str):
        self.colormap_scale_tag = StringValue(default_string = colormap_scale_tag)


    def print(self):
        print(f'{self.min_intensity = }')
        print(f'{self.max_intensity = }')
        print(f'{self.window_size = }')
        print(f'{self.colormap_name = }')
        print(f'{self.colormap_reversed = }')
        print(f'{self.colormap_scale_tag = }')
        print(f'{self.colormap_scale_type = }')

    def _cleanup_(self):
        attrib_list = list(self.__dict__.keys())
        while len(attrib_list) > 0:
            attrib_name = attrib_list.pop()
            try:
                getattr(self, attrib_name)._cleanup_()
            except:
                pass
            setattr(self, attrib_name, None)
            delattr(self, attrib_name)


class GeometryInfo(object):
    def __init__(self, 
                 default_geometry: list[float, float, float] = [1.0, 1.0, 1.0],
                 default_limits: list[float, float] = [0.25, 10.0]):
        
        self.pixel_spacing_x = OptionValue(tag = G.OPTION_TAG_DICT['pixel_spacing_x'], 
                                           default_value = 1.0*default_geometry[0], 
                                           default_limits = default_limits)
        self.pixel_spacing_y = OptionValue(tag = G.OPTION_TAG_DICT['pixel_spacing_y'], 
                                           default_value = 1.0*default_geometry[1], 
                                           default_limits = default_limits)
        self.slice_thickness = OptionValue(tag = G.OPTION_TAG_DICT['slice_thickness'], 
                                           default_value = 1.0*default_geometry[2], 
                                           default_limits = default_limits)


    def reset_geometry(self):
        self.pixel_spacing_x.reset()
        self.pixel_spacing_y.reset()
        self.slice_thickness.reset()
        

    def update_geometry(self):
        self.pixel_spacing_x.update_values(dpg.get_value(self.pixel_spacing_x.tag))
        self.pixel_spacing_y.update_values(dpg.get_value(self.pixel_spacing_y.tag))
        self.slice_thickness.update_values(dpg.get_value(self.slice_thickness.tag))

    
    def get_pixel_steps(self):
        return [self.slice_thickness.current_value, self.pixel_spacing_y.current_value, self.pixel_spacing_x.current_value]


    def __repr__(self):
        return f'Volume Geometry: \n\tPixel Spacing  : ({self.pixel_spacing_x}, {self.pixel_spacing_y})\n\tSlice Thickness: {self.slice_thickness}'


    def __call__(self):
        return [self.slice_thickness.current_value, self.pixel_spacing_y.current_value, self.pixel_spacing_x.current_value]
    

    def _cleanup_(self):
        attrib_list = list(self.__dict__.keys())
        while len(attrib_list) > 0:
            attrib_name = attrib_list.pop()
            try:
                getattr(self, attrib_name)._cleanup_()
            except:
                pass
            setattr(self, attrib_name, None)
            delattr(self, attrib_name)


class CameraInfo(object):
    def __init__(self, 
                 tag = 'Camera',
                 default_position = [0.0, 0.0, 0.0],
                 default_angle = [0.0, 0.0, 0.0],
                 default_distance = 50.0,
                 default_lens = 'flat',
                 default_norm = 0.0):
        
        pass

class OrientationInfo(object):
    def __init__(self, 
                 tag = 'Orientation',
                 default_origin: list[float] = [0.0, 0.0, 0.0],
                 default_angle: list[float] = [0.0, 0.0, 0.0],
                 default_norm: float = 0.0,
                 default_norm_sign: float = -1.0,
                 default_quaternion: qtn.QuaternionicArray = G.QTN_DICT[G.VIEW], 
                 default_geometry: list[float] = [1.0, 1.0, 1.0],
                 default_limits_origin: list[float] = [-500, 500],
                 default_limits_norm: list[float] = [-500, 500],
                 default_limits_angle: list[float] = [-180, 180],
                 default_limits_geometry: list[float] = [0.10, 50.0],
                 default_texture_dim: float = G.TEXTURE_DIM,
                 default_texture_center: float = G.TEXTURE_CENTER,
                 voxel_start: np.ndarray = np.zeros(3),
                 voxel_center: np.ndarray = np.zeros(3),
                 voxel_steps: np.ndarray = np.ones(3),
                 default_drawlayer_tags: list[str] = ['', '']):

        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()

        self.tag = StringValue(default_string=tag)
        self.origin_x = OptionValue(tag = G.OPTION_TAG_DICT['origin_x'], default_value = default_origin[0], default_limits = default_limits_origin)
        self.origin_y = OptionValue(tag = G.OPTION_TAG_DICT['origin_y'], default_value = default_origin[1], default_limits = default_limits_origin)
        self.origin_z = OptionValue(tag = G.OPTION_TAG_DICT['origin_z'], default_value = default_origin[2], default_limits = default_limits_origin)
        self.norm = OptionValue(tag = G.OPTION_TAG_DICT['norm'], default_value = default_norm, default_limits = default_limits_norm)
        self.pitch = OptionValue(tag = G.OPTION_TAG_DICT['pitch'], default_value = default_angle[0], default_limits = default_limits_angle)
        self.yaw = OptionValue(tag = G.OPTION_TAG_DICT['yaw'], default_value = default_angle[1], default_limits = default_limits_angle)
        self.roll = OptionValue(tag = G.OPTION_TAG_DICT['roll'], default_value = default_angle[2], default_limits = default_limits_angle)
        self.quaternion = QuaternionValue(tag = 'quaternion', default_quaternion = 1.0*default_quaternion)
        self.global_quaternion = QuaternionValue(tag = 'quaternion', default_quaternion = 1.0*default_quaternion)
        self.origin_vector = VectorValue(tag = 'origin_vector', default_vector = 1.0*cp.array([[0.0], [0.0], [0.0]]))
        self.scaled_origin_vector = VectorValue(tag = 'scaled_origin_vector', default_vector = 1.0*cp.array([[0.0], [0.0], [0.0]]))
        self.norm_vector = VectorValue(tag='norm_vector', default_vector = 1.0*cp.array([[0.0], [0.0], [default_norm_sign]])) # Start by looking along Z axis, Axial View.
        self.viewport_origin_vector = VectorValue(tag = 'viewport_origin', default_vector = 1.0*cp.array([[0.0], [0.0], [0.0]]))
        self.pixel_spacing_x = OptionValue(tag = G.OPTION_TAG_DICT['pixel_spacing_x'], default_value = 1.0*default_geometry[0], default_limits = default_limits_geometry, default_step = 0.1)
        self.pixel_spacing_y = OptionValue(tag = G.OPTION_TAG_DICT['pixel_spacing_y'], default_value = 1.0*default_geometry[1], default_limits = default_limits_geometry, default_step = 0.1)
        self.slice_thickness = OptionValue(tag = G.OPTION_TAG_DICT['slice_thickness'], default_value = 1.0*default_geometry[2], default_limits = default_limits_geometry, default_step = 0.1)
        self.geometry_vector = VectorValue(tag = 'geometry_vector', default_vector = cp.array([[1.0*default_geometry[0]], [1.0*default_geometry[1]], [1.0*default_geometry[2]]]))
        self.view_plane = ViewPlane(tag = f'{tag}|ViewPlane', texture_dim = default_texture_dim, texture_center = default_texture_center, 
                                    z_dim = 2, voxel_start = voxel_start, voxel_steps = voxel_steps, voxel_center = voxel_center)
        self.view_plane_ortho = ViewPlane(tag = f'{tag}|ViewPlaneOrtho', texture_dim = default_texture_dim, texture_center = default_texture_center, 
                                          z_dim = 1, voxel_start = voxel_start, voxel_steps = voxel_steps, voxel_center = voxel_center)
        self.drawlayer = StringValue(default_string = default_drawlayer_tags[0])
        self.drawlayer_ortho = StringValue(default_string = default_drawlayer_tags[1])
        self.volume_basis = np.eye(3, dtype = np.float32)

    def get_volume_coords(self, 
                          coord_x:int, 
                          coord_y:int):
        return self.quaternion.current_value.rotate(
                    np.array([[-1.0*coord_y],
                              [coord_x],
                              [0]]),
                              axis = 0).round(decimals = 4).reshape((3,))


    def set_drawlayer_tags(self, 
                          drawlayer_tags: list[str]):
        self.drawlayer = StringValue(default_string = drawlayer_tags[0])
        self.drawlayer_ortho = StringValue(default_string = drawlayer_tags[1])
    

    def get_drawlayer_tags(self) -> list[str]:

        return [self.drawlayer.current_value, self.drawlayer_ortho.current_value]
    
    
    def copy_orientation(self, 
                         orientation_info: "OrientationInfo"):
        
        for orientation_key in orientation_info.__dict__.keys():
            check_string = f'{orientation_key}\n--------------------\n'
            
            if orientation_key != 'view_plane':
                check_string = f'{check_string}{getattr(self, orientation_key)}\nNew: '
            
            for value_type in ['current_value', 'previous_value', 'difference_value']:
                setattr(getattr(self, orientation_key), value_type, getattr(getattr(orientation_info, orientation_key), value_type))

            if orientation_key != 'view_plane':
                check_string = f'{check_string}{getattr(self, orientation_key)}'

            if (orientation_key != 'view_plane') and G.DEBUG_MODE:
                print(check_string)


    def set_orientation(self, 
                        pitch:float = None, 
                        yaw:float = None, 
                        roll:float = None, 
                        quaternion = None,
                        norm:float = None, 
                        origin_x:float = None,
                        origin_y:float = None, 
                        origin_z:float = None,
                        pixel_spacing_x:float = None, 
                        pixel_spacing_y:float = None, 
                        slice_thickness:float = None,
                        drawlayer_tags:list[str] = None):
                        
            with dpg.mutex():
                if isinstance(quaternion, qtn.array):
                    self.set_quaternion(quaternion = quaternion * self.quaternion.default_value)

                else:
                    self.set_pitch_yaw_roll(pitch = pitch, 
                                            yaw = yaw,
                                            roll = roll)
                    self.set_quaternion()

                self.set_geometry(pixel_spacing_x = pixel_spacing_x,
                                  pixel_spacing_y = pixel_spacing_y,
                                  slice_thickness = slice_thickness)
                
                self.set_geometry_vector()
                self.set_norm(norm = norm)
                # We apply scaling here because we should be getting the positions in mm space, not in texture space. 
                self.set_origin(origin_x = origin_x,
                                origin_y = origin_y,
                                origin_z = origin_z)
                self.set_origin_vector()

                self.set_norm_vector()
                self.set_view_plane()
                

    def update_orientation(self, 
                           pitch:float = None, 
                           yaw:float = None, 
                           roll:float = None, 
                           norm:float = None, 
                           quaternion = None,
                           origin_x:float = None,
                           origin_y:float = None, 
                           origin_z:float = None,
                           pixel_spacing_x:float = None, 
                           pixel_spacing_y:float = None, 
                           slice_thickness:float = None,
                           drawlayer_tags:list[str] = None):
        with dpg.mutex():
            # Update geometry first, then apply to all other vectors.  
            # Reset View Plane Zoom. 
            self.view_plane.update_values(self.view_plane.current_value 
                                          * self.geometry_vector.current_value)
            self.view_plane_ortho.update_values(self.view_plane_ortho.current_value 
                                                * self.geometry_vector.current_value)
            
            self.set_pitch_yaw_roll(pitch = pitch, 
                                    yaw = yaw,
                                    roll = roll)
            
            self.update_quaternion()
            self.set_geometry(pixel_spacing_x = pixel_spacing_x,
                              pixel_spacing_y = pixel_spacing_y,
                              slice_thickness = slice_thickness)
            
            self.set_geometry_vector()
            self.set_norm(norm = norm)
            # We apply scaling here because we should be getting the positions in mm space, 
            # not in texture space. 
            self.set_origin(origin_x = origin_x,
                            origin_y = origin_y,
                            origin_z = origin_z)
            
            # Update vectors and view plane
            self.update_norm_vector()
            # self.update_viewport_origin_vector(scale_vector = self.geometry_vector.get_quotient_value(),
            #                                    apply_scaling = False)
            
            self.set_origin_vector()
            self.update_view_plane()

            # Set Zoom Plane and origin values zoom.
            self.view_plane.update_values(self.view_plane.current_value 
                                          / self.geometry_vector.current_value)
            self.view_plane_ortho.update_values(self.view_plane_ortho.current_value 
                                                / self.geometry_vector.current_value)
            
            # # We reset scaling here so that the origin_x, origin_y, and origin_z 
            # values reflect correct the OptionsPanel values. 
            # self.origin_x.set_current_value(self.origin_x.current_value 
            #                                 / self.geometry_vector.current_value.reshape((3, )).get()[1])
            # self.origin_y.set_current_value(self.origin_y.current_value 
            #                                 / self.geometry_vector.current_value.reshape((3, )).get()[0])
            # self.origin_z.set_current_value(self.origin_z.current_value 
            #                                 / self.geometry_vector.current_value.reshape((3, )).get()[2])

            if drawlayer_tags == None:
                drawlayer_tags = self.get_drawlayer_tags()
            self.update_drawlayers(drawlayer_tags)

            self.volume_basis = self.quaternion.current_value.rotate(np.eye(3, dtype=np.float32), axis = 0)

            print(f'update_orientation')
            print(f'\tVolume Basis: {self.volume_basis}')

    def update_drawlayers(self, 
                          drawlayer_tags: list[str]):
        self.drawlayer.update_values(drawlayer_tags[0])
        self.drawlayer_ortho.update_values(drawlayer_tags[1])
    

    def reset_origin(self):
        with dpg.mutex():
            self.norm.reset()
            self.origin_x.reset()
            self.origin_y.reset()
            self.origin_z.reset()
            self.viewport_origin_vector.reset()
            self.view_plane.update_values(self.view_plane.current_value - self.scaled_origin_vector.current_value)
            self.view_plane_ortho.update_values(self.view_plane_ortho.current_value - self.scaled_origin_vector.current_value)
            self.origin_vector.reset()
            self.scaled_origin_vector.reset()
            self.norm_vector.reset()


    def reset_angle(self):
        with dpg.mutex():
            self.pitch.reset()
            self.yaw.reset()
            self.roll.reset()

            self.view_plane.update_values(qtn_rotate(self.quaternion.current_value.inverse,
                                                     self.view_plane.current_value - self.scaled_origin_vector.current_value, 
                                                     axis = 0)
                                                + self.scaled_origin_vector.current_value,
                                                decimals = 4)
            self.view_plane_ortho.update_values(qtn_rotate(self.quaternion.current_value.inverse,
                                                           self.view_plane_ortho.current_value - self.scaled_origin_vector.current_value, 
                                                           axis = 0)
                                                        + self.scaled_origin_vector.current_value,
                                                        decimals = 4)
            self.norm_vector.update_values(qtn_rotate(self.quaternion.current_value.inverse,
                                                      self.norm_vector.current_value,
                                                      axis = 0),
                                                    decimals=4)
            
            self.quaternion.reset()
            self.global_quaternion.reset()


    def reset_geometry(self):
        with dpg.mutex():
            self.pixel_spacing_x.reset()
            self.pixel_spacing_y.reset()
            self.slice_thickness.reset()
            self.view_plane.update_values(self.view_plane.current_value * self.geometry_vector.current_value)
            self.view_plane_ortho.update_values(self.view_plane_ortho.current_value * self.geometry_vector.current_value)

            self.scaled_origin_vector.update_values(self.origin_vector.current_value)
            self.geometry_vector.reset()


    def reset_orientation(self):
        with dpg.mutex():
            self.pitch.reset()
            self.yaw.reset()
            self.roll.reset()
            self.pixel_spacing_x.reset()
            self.pixel_spacing_y.reset()
            self.slice_thickness.reset()
            self.geometry_vector.reset()
            self.norm.reset()
            self.origin_x.reset()
            self.origin_y.reset()
            self.origin_z.reset()
            self.quaternion.reset()
            self.global_quaternion.reset()
            self.viewport_origin_vector.reset()
            self.origin_vector.reset()
            self.scaled_origin_vector.reset()
            self.norm_vector.reset()
            self.view_plane.reset()
            self.view_plane_ortho.reset()
        
          
    def set_pitch_yaw_roll(self, 
                           pitch:float = None, 
                           yaw:float = None,
                           roll:float = None): #dpg.get_value(slider_tag)
        
        """
        To maintain patient geometry on screen: 

        j_hat   ->      Anterior-Posterior  Positive towards Posterior
        k_hat   ->      Inferior-Superior   Positive towards Superior
        i_hat   ->      Left-Right          Positive towards Left

        yaw     ->      About j_hat         Positive i_hat -> k_hat; head right, feet left
        pitch   ->      About i_hat         Positive j_hat -> k_hat; head foward, feet back
        roll    ->      About k_hat         Positive i_hat -> j_hat; Roll left

        """

        if yaw == None:
            yaw = dpg.get_value(G.OPTION_TAG_DICT['yaw'])
        if pitch == None:
            pitch = dpg.get_value(G.OPTION_TAG_DICT['pitch'])
        if roll == None:
            roll = dpg.get_value(G.OPTION_TAG_DICT['roll'])

        self.pitch.update_values(pitch)
        self.yaw.update_values(-1.0 * yaw) # We need to do this since we are setting our norm plane to be [0.0, 0.0, -1.0]
        self.roll.update_values(roll)
        
    def set_quaternion(self, quaternion: qtn.QuaternionicArray):

        if isinstance(quaternion, qtn.array):
            self.quaternion.update_values(quaternion)
            self.global_quaternion.update_values(quaternion)

            return
        
        yaw_rtn = qtn.array.from_axis_angle([np.deg2rad(self.yaw.current_value), 0.0, 0.0])
        pitch_rtn = qtn.array.from_axis_angle([0.0, np.deg2rad(self.pitch.current_value), 0.0])
        roll_qtn = qtn.array.from_axis_angle([0.0, 0.0, np.deg2rad(self.roll.current_value)])
        
        self.global_quaternion.update_values(roll_qtn * pitch_rtn * yaw_rtn, decimals = 6)
        self.quaternion.update_values(roll_qtn * pitch_rtn * yaw_rtn, decimals = 6)
        
    def update_quaternion(self):
        """
        To update our current quaternion, we multiply it by the difference between the new and previous value. 
        
        yaw     -> rotate about j_hat
        pitch   -> rotate about i_hat
        roll    -> rotate about k_hat

        """

        yaw_rtn = qtn.array.from_axis_angle([np.deg2rad(self.yaw.accumulated_value), 0.0, 0.0])
        pitch_rtn = qtn.array.from_axis_angle([0.0, np.deg2rad(self.pitch.accumulated_value), 0.0])
        roll_qtn = qtn.array.from_axis_angle([0.0, 0.0, np.deg2rad(self.roll.accumulated_value)])

        # yaw_rtn = qtn.array.from_axis_angle([np.deg2rad(self.yaw.current_value), 0.0, 0.0])
        # pitch_rtn = qtn.array.from_axis_angle([0.0, np.deg2rad(self.pitch.current_value), 0.0])
        # roll_qtn = qtn.array.from_axis_angle([0.0, 0.0, np.deg2rad(self.roll.current_value)])
        
        self.global_quaternion.update_values(roll_qtn * pitch_rtn * yaw_rtn * self.global_quaternion.default_value, decimals = 6)

        rotate_qtn = qtn.array([1.0, 0.0, 0.0, 0.0])

        for angle_index, angle_tag in enumerate(['yaw', 'pitch', 'roll']):
            angle: QuaternionValue = getattr(self, angle_tag)
            temp_qtn = qtn.array([1.0, 0.0, 0.0, 0.0])
            if angle.difference_value != 0:
                angle_rad = np.deg2rad(angle.difference_value)
                temp_qtn[0] = np.round(np.cos(angle_rad/2), decimals = 9)
                temp_qtn[angle_index + 1] = np.round(np.sin(angle_rad/2), decimals = 9)
                rotate_qtn = temp_qtn * rotate_qtn

        self.quaternion.update_values(rotate_qtn * self.quaternion.current_value, decimals = 6)

        print('OptionValue Message: update_quaternion')
        print(f'{self.global_quaternion.current_value * self.quaternion.current_value.inverse}')

        
    def set_norm(self, 
                    norm:float = None):
        if norm == None:
            norm = dpg.get_value(G.OPTION_TAG_DICT['norm'])
        self.norm.update_values(norm)
        
        
    def set_origin(self, 
                   origin_x:float = None, 
                   origin_y:float = None,
                   origin_z:float = None):
        
        if origin_x == None:
            origin_x = dpg.get_value(G.OPTION_TAG_DICT['origin_x'])
        if origin_y == None: 
            origin_y = dpg.get_value(G.OPTION_TAG_DICT['origin_y'])
        if origin_z == None: 
            origin_z = dpg.get_value(G.OPTION_TAG_DICT['origin_z'])

        self.origin_x.update_values(origin_x) # Make these pix/mm. We choose our location using mm. 
        self.origin_y.update_values(origin_y) # When we set origin_x = 50, we want to stay there. 
        self.origin_z.update_values(origin_z) # However, this now makes the increments off by a 
                                              # the same factor. Pushing it moves it multiple pixels. 
        

    def set_geometry(self, 
                     pixel_spacing_x:float = None,
                     pixel_spacing_y:float = None,
                     slice_thickness:float = None):
        
        if pixel_spacing_x == None:
            pixel_spacing_x = dpg.get_value(self.pixel_spacing_x.tag)
        if pixel_spacing_y == None:
            pixel_spacing_y = dpg.get_value(self.pixel_spacing_y.tag)
        if slice_thickness == None:
            slice_thickness = dpg.get_value(self.slice_thickness.tag)

        self.pixel_spacing_x.update_values(pixel_spacing_x, decimals = 4)
        self.pixel_spacing_y.update_values(pixel_spacing_y, decimals = 4)
        self.slice_thickness.update_values(slice_thickness, decimals = 4)


    def set_geometry_vector(self) -> None:
        self.geometry_vector.update_values(
                       cp.array([[self.pixel_spacing_y.current_value],
                                 [self.pixel_spacing_x.current_value], 
                                 [self.slice_thickness.current_value]],
                                 dtype = cp.float32),
                                 decimals = 4)


    def set_origin_vector(self, apply_scaling = False):
        """
        
        Increments origin_vector by applying the appropriate rotations and translations: 

            V_O_c + Q_C( Q_D( [N_delta, -1.0*O_y_delta, O_x_delta] ) )

        where:
            V_O_c is        origin_vector.current_value
            N_delta is      norm.difference_value
            O_y_delta is    origin_y.difference_value
            O_x_delta is    origin_x.difference_value
            Q_C is          quaternion.current_value
            Q_D is          quaternion.default_value

        We use this version because we want to know which direction we need to step in for our origin, 
        given the Slice Location slider, Origin X slider, and Origin Y slider. 

        First we rotate the difference in the origin components by the default rotation. 
        Then we rotate that by the current quaternion.
        Then we add that to the current origin vector. 
        """

        
        self.origin_vector.update_values(
                            cp.array([[-1.0*self.origin_y.current_value],
                                      [self.origin_x.current_value],
                                      [self.origin_z.current_value]],
                                      dtype = cp.float32),
                                    decimals = 4
                                    )
        
        self.scaled_origin_vector.update_values(
                            cp.array([[-1.0*self.origin_y.current_value * self.pixel_spacing_y.current_value],
                                      [self.origin_x.current_value * self.pixel_spacing_x.current_value],
                                      [self.origin_z.current_value * self.slice_thickness.current_value]],
                                      dtype = cp.float32),
                                    decimals = 4
                                    )
        

    def set_norm_vector(self):
        self.norm_vector.update_values(
            qtn_rotate(self.quaternion.current_value,
                       self.norm_vector.current_value, 
                       axis = 0), 
                       decimals = 4)
        

    def update_norm_vector(self):
        self.norm_vector.update_values(
            qtn_rotate(self.quaternion.difference_value,
                       self.norm_vector.current_value, 
                       axis = 0), 
                       decimals = 4)


    def update_viewport_origin_vector(self,
                                      scale_vector:np.ndarray|cp.ndarray = np.ones((3, 1)),
                                      apply_scaling:bool = False):
        
        if apply_scaling:
            self.viewport_origin_vector.update_values(self.viewport_origin_vector.current_value / scale_vector,
                                                      decimals = 4)

        self.viewport_origin_vector.update_values(
            self.viewport_origin_vector.current_value + 
            qtn_rotate(self.quaternion.current_value,
                    cp.array([[-1.0*self.origin_y.difference_value], 
                              [1.0*self.origin_x.difference_value], 
                              [0]]),
                axis = 0), 
                decimals = 4
            )

    def set_view_plane(self):
        if G.GPU_MODE:
            print('OptionValue set_view_plane: Center of View Plane Init')
            print(f'\t{self.view_plane.get_coords(self.view_plane.texture_center, self.view_plane.texture_center)}')

            self.view_plane.update_values(self.view_plane.default_value - self.scaled_origin_vector.current_value)
            print('OptionValue set_view_plane: Center of View Plane Post Shift')
            print(f'\t{self.view_plane.get_coords(self.view_plane.texture_center, self.view_plane.texture_center)}')

            self.view_plane.update_values(
                qtn_rotate(self.quaternion.current_value,
                           self.view_plane.current_value, axis = 0 )
                           )
            print('OptionValue set_view_plane: Center of View Plane Post Rotation')
            print(f'\t{self.view_plane.get_coords(self.view_plane.texture_center, self.view_plane.texture_center)}')

            self.view_plane.update_values(self.view_plane.current_value + self.scaled_origin_vector.current_value + self.scaled_origin_vector.current_value)
            print('OptionValue set_view_plane: Center of View Plane Post Reshift')
            print(f'\t{self.view_plane.get_coords(self.view_plane.texture_center, self.view_plane.texture_center)}')

            self.view_plane.update_values(self.view_plane.current_value / self.geometry_vector.current_value)
            print('OptionValue set_view_plane: Center of View Plane Post Scale')
            print(f'\t{self.view_plane.get_coords(self.view_plane.texture_center, self.view_plane.texture_center)}')
            
            
            self.view_plane_ortho.update_values(
                ((qtn_rotate(self.quaternion.current_value,
                             self.view_plane_ortho.default_value
                             - self.origin_vector.current_value, axis = 0)
                        + self.origin_vector.current_value
                        + self.origin_vector.current_value)
                    / self.geometry_vector.current_value), 
                        decimals = 4)
            
        else:
            print('GPU required!')
            return
    
    def update_view_plane(self):
        """
        The math for this goes: 
            (diff_qtn.rotate(view_plane.current_value 
                             - origin_vector.current_value)
            + origin_vector.current_value 
            + origin_vector.difference_value 
            + self.norm.difference_value*self.norm_vector.current_value)

        Which moves the view plane to the origin, rotates it, then moves it back to center. 
        Then we add the difference between our current step and the previous to move the origin.
        Then we add the differnce in our norm to move our camera viewpoint. 
        
        It's convoluted because we control the orientation from the viewplane space, 
        transform it into volume space, and then back into viewplane space

        """
        if G.GPU_MODE:
            self.view_plane.update_values(
                (qtn_rotate(self.quaternion.difference_value,
                            self.view_plane.current_value
                            - self.scaled_origin_vector.current_value, axis = 0)
                        + self.scaled_origin_vector.current_value
                        + self.scaled_origin_vector.difference_value), 
                        decimals = 4)
            
            self.view_plane_ortho.update_values(
                (qtn_rotate(self.quaternion.difference_value,
                            self.view_plane_ortho.current_value
                            - self.scaled_origin_vector.current_value, axis = 0)
                        + self.scaled_origin_vector.current_value
                        + self.scaled_origin_vector.difference_value), 
                        decimals = 4)
            
        else:
            print('GPU required!')
            return
    
    
    def _cleanup_(self):
        attrib_list = list(self.__dict__.keys())
        while len(attrib_list) > 0:
            attrib_name = attrib_list.pop()
            try:
                getattr(self, attrib_name)._cleanup_()
            except:
                pass
            setattr(self, attrib_name, None)
            delattr(self, attrib_name)

    def print_values(self):
        print('OptionValue Message: Parameter Info:')
        print(f'{"-"*150}')
        print(f'{self.quaternion.current_value = }')
        print(f'{self.quaternion.previous_value = }')
        print(f'{self.quaternion.difference_value = }')
        print(f'{self.origin_x.current_value = }')
        print(f'{self.origin_x.previous_value = }')
        print(f'{self.origin_x.difference_value = }')
        print(f'{self.origin_y.current_value = }')
        print(f'{self.origin_y.previous_value = }')
        print(f'{self.origin_y.difference_value = }')
        print(f'{self.origin_z.current_value = }')
        print(f'{self.origin_z.previous_value = }')
        print(f'{self.origin_z.difference_value = }')
        print(f'{self.geometry_vector.pretty_print(modifier = "current_value") = }')
        print(f'{self.geometry_vector.pretty_print(modifier = "previous_value") = }')
        print(f'{self.geometry_vector.pretty_print(modifier = "difference_value") = }')
        print(f'{self.geometry_vector.pretty_print(modifier = "get_quotient_value()") = }')
        print(f'{self.origin_vector.pretty_print(modifier = "current_value") = }')
        print(f'{self.origin_vector.pretty_print(modifier = "previous_value") = }')
        print(f'{self.origin_vector.pretty_print(modifier = "difference_value") = }')
        print(f'{self.scaled_origin_vector.pretty_print(modifier = "current_value") = }')
        print(f'{self.scaled_origin_vector.pretty_print(modifier = "previous_value") = }')
        print(f'{self.scaled_origin_vector.pretty_print(modifier = "difference_value") = }')
        print(f'{self.viewport_origin_vector.pretty_print(modifier = "current_value") = }')
        print(f'{self.viewport_origin_vector.pretty_print(modifier = "previous_value") = }')
        print(f'{self.viewport_origin_vector.pretty_print(modifier = "difference_value") = }')
        print(f'{self.norm_vector.pretty_print(modifier = "current_value") = }')
        print(f'{self.norm_vector.pretty_print(modifier = "previous_value") = }')
        print(f'{self.norm_vector.pretty_print(modifier = "difference_value") = }')
        print(f'{self.norm.current_value = }')
        print(f'{self.norm.previous_value = }')
        print(f'{self.norm.difference_value = }')
        print(f'{"-"*150}')

class HistogramInfo(object):
    def __init__(self, default_histogram = [0.5, 1200.5, 1, 'Linear']): # [min, max, step, scale]
        self.bins_min = OptionValue(tag = 'histogram_bins_min', default_value = default_histogram[0])
        self.bins_max = OptionValue(tag = 'histogram_bins_max', default_value = default_histogram[1])
        self.bin_step = OptionValue(tag = 'histogram_bin_step', default_value = default_histogram[2])
        self.bin_edges = ArrayValue(tag = 'histogram_bin_edges', default_array = cp.arange(self.bins_min(), 
                                                                                           self.bins_max(), 
                                                                                           self.bin_step()))
        self.bin_centers = ArrayValue(tag = 'histogram_bin_centers', default_array = cp.arange(self.bins_min(), 
                                                                                               self.bins_max(), 
                                                                                               self.bin_step()))
        self.bin_counts = ArrayValue(tag = 'histogram_bin_counts', default_array = cp.ones(self.bin_centers().shape))
        self.count_scale = StringValue(tag = 'histogram_count_scale', default_string = default_histogram[3])

        self.update_histogram_params(min_value = self.bins_min(), 
                                     max_value = self.bins_max(), 
                                     bin_step = self.bin_step())
    
    
    def return_size(self):
        return ['bin_centers', self.bin_centers.current_value.nbytes,
                'bin_edges', self.bin_centers.current_value.nbytes,
                'bin_counts', self.bin_counts.current_value.nbytes]


    def update_histogram_params(self, 
                                min_value = None, 
                                max_value = None, 
                                bin_step = None):
        
        self.update_bins(min_value = min_value, 
                         max_value = max_value, 
                         bin_step = bin_step)
        self.update_scale()


    def update_bins(self, 
                    min_value = None, 
                    max_value = None,
                    bin_step = None):

        self.bins_min.update_values(self.bins_min.default_value if min_value is None else min_value)
        self.bins_max.update_values(self.bins_max.default_value if max_value is None else max_value)
        self.bin_step.update_values(self.bin_step.default_value if bin_step is None else bin_step)
        self.bin_edges.update_values(cp.arange(self.bins_min(), self.bins_max() + self.bin_step(), self.bin_step()))
        self.bin_centers.update_values(self.bin_edges()[:-1] + (self.bin_edges()[1:] - self.bin_edges()[:-1])/2.0)

    #TODO Complete update_scale function. 
    def update_scale(self):
        # self.count_scale = f'{dpg.get_value("InfoBoxTab_histogram_scale")}'
        pass


    def update_histogram_counts(self, histogram_data, density = True):

        self.bin_counts.update_values(cp.histogram(histogram_data, bins = self.bin_edges())[0])


    def get_histogram(self, return_order = 'reversed', count_scale = 'percentage'):
        if return_order == 'reversed':
            return [cp_to_np(self.bin_centers.current_value), 
                    cp_to_np(self.bin_counts.current_value / cp.sum(self.bin_counts.current_value))]
        else:
            return [cp_to_np(self.bin_counts.current_value), 
                    cp_to_np(self.bin_centers.current_value / cp.sum(self.bin_counts.current_value))]


    def reset_histogram(self):
        self.bins_min.reset()
        self.bins_max.reset()
        self.bin_step.reset()
        self.count_scale = f'{self.default_scale}'
        self.bin_centers = None

    
    def copy_histogram(self, histogram_info: "HistogramInfo"):
        for histogram_key in histogram_info.__dict__.keys():
            check_string = f'{histogram_key}\n--------------------\n'
            check_string = f'{check_string}{getattr(self, histogram_key)}\nNew: '
            for value_type in ['current_value', 'previous_value']:
                if histogram_key == 'count_scale':
                    setattr(getattr(self, histogram_key), value_type, f'{getattr(getattr(histogram_info, histogram_key), value_type)}')
                else:
                   setattr(getattr(self, histogram_key), value_type, 1.0*getattr(getattr(histogram_info, histogram_key), value_type))

            if G.DEBUG_MODE:
                print(check_string)

    def get_info(self) -> dict:
        return dict(self.__dict__)

    def _cleanup_(self):
        attrib_list = list(self.__dict__.keys())
        while len(attrib_list) > 0:
            attrib_name = attrib_list.pop()
            try:
                getattr(self, attrib_name)._cleanup_()
            except:
                pass
            setattr(self, attrib_name, None)
            delattr(self, attrib_name)


class StringValue(object):
    def __init__(self, tag = '', default_string:str = 'Default'):

        self.tag = tag
        self.default_value:str = f'{default_string}'
        self.current_value:str = f'{default_string}'
        self.difference_value:str = f'{default_string}'
        self.previous_value:str = f'{default_string}'

    def __repr__(self):
        return f'{self.current_value}'

    def __call__(self):
        return f'{self.current_value}'
        
    def set_previous_value(self):
        self.previous_value = f'{self.current_value}'
    
    def set_current_value(self, new_current):
        self.current_value = f'{new_current}'
        
    def update_values(self, new_current):
        self.set_previous_value()
        self.set_current_value(new_current)
        
    def reset(self):
        self.set_current_value(self.default_value)
        self.set_previous_value()

    def set_info(self, info_dict: dict):
        self.__dict__ = dict(info_dict)
        
    def get_info(self) -> dict:
        return dict(self.__dict__)

    def _cleanup_(self):
        attrib_list = list(self.__dict__.keys())
        while len(attrib_list) > 0:
            attrib_key = attrib_list.pop()
            setattr(self, attrib_key, None)
            delattr(self, attrib_key)


class ColormapValue(object):
    def __init__(self, 
                 tag:str = 'colormap_combo',
                 default_colormap: list[interp1d] = G.COLORMAP_DICT['Fire']):
        self.tag = tag
        self.default_value: list[interp1d] = default_colormap
        self.current_value: list[interp1d] = default_colormap
        self.difference_value: list[interp1d] = default_colormap
        self.previous_value: list[interp1d] = default_colormap

    def __call__(self):
        return self.current_value
    
    def set_previous_value(self):
        self.previous_value = self.current_value

    def set_current_value(self, new_current: list[interp1d]):
        self.current_value = new_current

    def update_values(self, new_current: list[interp1d]):
        self.set_previous_value()
        self.set_current_value(new_current)

    def reset(self):
        self.set_current_value(self.default_value)
        self.set_previous_value()

    def _cleanup_(self):
        attrib_list = list(self.__dict__.keys())
        while len(attrib_list) > 0:
            attrib_key = attrib_list.pop()
            setattr(self, attrib_key, None)
            delattr(self, attrib_key)


class ArrayValue(object):
    def __init__(self, 
                 tag:str = '', 
                 default_array:np.ndarray|cp.ndarray = np_to_cp(np.zeros((5, 5, 5))),
                 default_steps:np.ndarray|cp.ndarray = np.ones((3))
                 ):
        
        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()

        self.tag:str = tag
        self.default_value:np.ndarray|cp.ndarray = 1.0*default_array
        self.current_value:np.ndarray|cp.ndarray = 1.0*default_array
        self.previous_value:np.ndarray|cp.ndarray = 1.0*default_array
        self.min_value:float|int = self.current_value.min()
        self.max_value:float|int = self.current_value.max()
        self.step_value: np.ndarray|cp.ndarray = 1.0*default_steps
    
    def __repr__(self):
        return f'{self.current_value}'

    def __call__(self):
        return self.current_value
        
    def set_previous_value(self):
        self.previous_value = 1.0*self.current_value
    
    def set_current_value(self, 
                          new_current:np.ndarray|cp.ndarray):
        self.current_value = 1.0*new_current
        
    def update_values(self, 
                      new_current:np.ndarray|cp.ndarray, 
                      decimals:int = 0):
        if decimals > 0:
            new_current = new_current.round(decimals = decimals)
        self.set_previous_value()
        self.set_current_value(new_current)
        
    def reset(self):
        self.set_current_value(1.0*self.default_value)
        self.set_previous_value()

    def set_info(self, info_dict: dict):
        self.__dict__ = dict(info_dict)
        
    def get_info(self) -> dict:
        return dict(self.__dict__)

    def _cleanup_(self):
        attrib_list = list(self.__dict__.keys())
        while len(attrib_list) > 0:
            attrib_key = attrib_list.pop()
            setattr(self, attrib_key, None)
            delattr(self, attrib_key)


class QuaternionValue(object):
    def __init__(self, tag = '', 
                 default_quaternion:qtn.QuaternionicArray = G.QTN_DICT[G.VIEW]):
    
        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()
        self.tag:str = tag
        self.default_value: qtn.QuaternionicArray = 1.0*default_quaternion
        self.current_value: qtn.QuaternionicArray = 1.0*default_quaternion
        self.previous_value: qtn.QuaternionicArray = 1.0*default_quaternion
        self.difference_value: qtn.QuaternionicArray = 1.0*default_quaternion


    def pretty_print(self):
        out_string = ''
        for value in self.current_value.round(2).squeeze(): # Squeeze it in case we have a (N, 1) vector
            out_string = f'{out_string}, {value + 0.0:.2f}'
        out_string = f'({out_string[2:]})'
        return out_string


    def __repr__(self):
        out_string = f'{self.current_value.scalar.round(3) + 0.0:.3f}'
        for qtn_element in ['i', 'j', 'k']:
            out_string = f'{out_string}, {getattr(self.current_value, qtn_element).round(3) + 0.0:.3f}'
        out_string = f'({out_string})'
        return out_string
    
    def __call__(self):
        return self.current_value
        
    def set_difference_value(self):
        self.difference_value = (self.current_value * self.previous_value.inverse)
        
    def set_previous_value(self):
        self.previous_value = 1.0*self.current_value
    
    def set_current_value(self, 
                          new_current:qtn.QuaternionicArray):
        self.current_value = 1.0*new_current
        
    def update_values(self, 
                      new_current:qtn.QuaternionicArray,
                      decimals:int = 0):
        if decimals > 0:
            new_current = np.round(new_current, decimals = decimals)
        self.set_previous_value()
        self.set_current_value(new_current)
        self.set_difference_value()
        
    def reset(self):
        self.set_current_value(1.0*self.default_value)
        self.set_previous_value()
        self.set_difference_value()

    def set_info(self, info_dict: dict):
        self.__dict__ = dict(info_dict)
        
    def get_info(self) -> dict:
        return dict(self.__dict__)

    def _cleanup_(self):
        attrib_list = list(self.__dict__.keys())
        while len(attrib_list) > 0:
            attrib_key = attrib_list.pop()
            setattr(self, attrib_key, None)
            delattr(self, attrib_key)


class VectorValue(object):
    def __init__(self, 
                 tag = '', 
                 default_vector = np.ones((3)),
                 default_steps = np.ones((3))):
        
        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()

        self.tag = tag
        self.default_value: np.ndarray|cp.ndarray = 1.0*default_vector
        self.current_value: np.ndarray|cp.ndarray = 1.0*default_vector
        self.previous_value: np.ndarray|cp.ndarray = 1.0*default_vector
        self.difference_value: np.ndarray|cp.ndarray = self.current_value - self.previous_value
        self.accumulated_value: np.ndarray|cp.ndarray = 0.0*default_vector
        self.step_value: np.ndarray|cp.ndarray = 1.0*default_steps


    def pretty_print(self, 
                     modifier = 'current_value'):
        out_string = ''

        if modifier == 'get_quotient_value()':
            for value in self.get_quotient_value().round(2).squeeze(): # Squeeze it in case we have a (N, 1) vector
                out_string = f'{out_string}, {value + 0.0:.2f}'
            out_string = f'({out_string[2:]})'
            return out_string
        
        if modifier == '':
            modifier = 'current_value'

        for value in getattr(self, modifier).round(2).squeeze(): # Squeeze it in case we have a (N, 1) vector
            out_string = f'{out_string}, {value + 0.0:.2f}'
        out_string = f'({out_string[2:]})'
        return out_string

    def __repr__(self):
        out_string = ''
        for value in self.current_value.round(2).squeeze(): # Squeeze it in case we have a (N, 1) vector
            out_string = f'{out_string}, {value + 0.0:.2f}'
        out_string = f'({out_string[2:]})'
        return out_string

    def __call__(self):
        return self.current_value
    
    def set_step_value(self,
                       new_step: np.ndarray|cp.ndarray):
        self.step_value = 1.0*new_step
    
    def get_quotient_value(self):
        "We want to know what fraction to change compared to the previous value"
        return self.current_value / self.previous_value

    def set_difference_value(self):
        self.difference_value = self.current_value - self.previous_value
        
    def set_previous_value(self):
        self.previous_value = 1.0*self.current_value

    def set_accumulated_value(self, value = None):
        if isinstance(value, type(None)):
            self.accumulated_value += self.difference_value
        else:
            self.accumulated_value = 1.0 * value
    
    def set_current_value(self, 
                          new_current:np.ndarray|cp.ndarray):
        self.current_value = 1.0*new_current
        
    def update_values(self, 
                      new_current:np.ndarray|cp.ndarray,
                      decimals:int = 0):
        if decimals > 0:
            new_current = new_current.round(decimals = decimals)
        self.set_previous_value()
        self.set_current_value(new_current)
        self.set_difference_value()
        self.set_accumulated_value()

    def rotate(self, quaternion: qtn.QuaternionicArray = qtn.array([1, 0, 0, 0])):
        self.update_values(quaternion.rotate(self.current_value, axis = 0))
        
    def reset(self):
        self.set_current_value(1.0*self.default_value)
        self.set_previous_value()
        self.set_difference_value()
        self.set_accumulated_value(0.0 * self.default_value)

    def set_info(self, info_dict: dict):
        self.__dict__ = dict(info_dict)
        
    def get_info(self) -> dict:
        return dict(self.__dict__)

    def _cleanup_(self):
        attrib_list = list(self.__dict__.keys())
        while len(attrib_list) > 0:
            attrib_key = attrib_list.pop()
            setattr(self, attrib_key, None)
            delattr(self, attrib_key)


class OptionValue(object):
    def __init__(self, 
                 tag:str = '', 
                 default_value:int|float = 0, 
                 default_limits:list[int|float] = [0, 3500],
                 default_step:int|float = 1.0):

        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()

        self.tag = tag
        self.default_value:int|float = 1.0*default_value
        self.previous_value:int|float = 1.0*default_value
        self.current_value:int|float = 1.0*default_value
        self.difference_value:int|float = self.current_value - self.previous_value 
        self.accumulated_value:int|float = 0.0
        self.default_step:int|float = 1.0*default_step
        self.step_value:int|float = 1.0*default_step
        self.step_fast_value:int|float = 5.0*default_step
        self.default_limits:list[int|float] = default_limits.copy()
        self.limit_low:int|float = default_limits[0]
        self.limit_high:int|float = default_limits[1]

    def __repr__(self):
        return f'{self.current_value}'
    
    def __call__(self):
        try:
            return self.current_value.item()
        except:
            return self.current_value
        
    def set_accumulated_value(self, value = None):
        if isinstance(value, type(None)):
            self.accumulated_value += self.difference_value
        else:
            self.accumulated_value = 1.0 * value
     
    def set_difference_value(self):
        self.difference_value = self.current_value - self.previous_value + 0.0
        
    def set_previous_value(self):
        self.previous_value = 1.0*self.current_value + 0.0
        
    def set_current_value(self, 
                          new_value:int|float):
        self.current_value = 1.0*new_value + 0.0

    def set_limits(self, 
                   new_limits:list[int|float]):
        self.limit_low = 1.0*new_limits[0] + 0.0
        self.limit_high = 1.0*new_limits[1] + 0.0
        
    def set_step_values(self, 
                        new_step_value:float|int):
        self.step_value = 1.0*new_step_value + 0.0
        self.step_fast_value = 5.0*new_step_value + 0.0
        
    def update_values(self, 
                      new_current:int|float,
                      decimals:int = 0):
        if decimals > 0:
            new_current = round(new_current, ndigits = decimals)
        self.set_previous_value()
        self.set_current_value(new_current)
        self.set_difference_value()
        self.set_accumulated_value()
        
    def reset(self):
        self.set_current_value(1.0*self.default_value)
        self.set_previous_value()
        self.set_difference_value()
        self.set_accumulated_value(value = 0.0)
        self.set_limits(self.default_limits)
        self.set_step_values(self.default_step)

    def set_info(self, info_dict: dict):
        self.__dict__ = dict(info_dict)

    def get_info(self) -> dict:
        return dict(self.__dict__)

    def _cleanup_(self):
        attrib_list = list(self.__dict__.keys())
        while len(attrib_list) > 0:
            attrib_key = attrib_list.pop()
            setattr(self, attrib_key, None)
            delattr(self, attrib_key)


class ViewPlane(object):
    """
    
    This is a view plane object. It holds position information only, which is then used to find the interpolated intensity
    values in a given volume. 

    """
    def __init__(self, 
                 tag:str = '', 
                 texture_dim:int = G.TEXTURE_DIM,
                 texture_center:int = G.TEXTURE_CENTER,
                 voxel_start: np.ndarray = np.zeros(3, dtype = np.float32),
                 voxel_center: np.ndarray = np.zeros(3, dtype = np.float32),
                 voxel_steps: np.ndarray = np.ones(3, dtype = np.float32),
                 z_dim:int = 2):

        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()
        
        self.tag = tag
        self.z_dim = z_dim
        self.texture_dim = texture_dim
        self.texture_center = texture_center
        self.voxel_start = 1.0*voxel_start.reshape((3, 1))
        self.voxel_center = 1.0*voxel_center.reshape((3, 1))
        self.voxel_steps = 1.0*voxel_steps.reshape((3, 1))

        self.dim_list = [0, 1, 2]
        self.dim_list.remove(self.z_dim)
        self.shape = (3, self.texture_dim, self.texture_dim)

        if G.GPU_MODE:
            self.default_value = cp.zeros((3, self.texture_dim, self.texture_dim), dtype = cp.float32)
            self.default_value[self.dim_list] = cp.indices((self.texture_dim, self.texture_dim), dtype = cp.float32) - self.texture_center
            self.default_value = self.default_value.reshape((3, self.texture_dim*self.texture_dim))
            self.default_value = qtn_rotate(G.QTN_DICT[G.VIEW], self.default_value, axis = 0)
    
            self.voxel_start:cp.ndarray = cp.array(voxel_start).reshape((3, 1))
            self.voxel_center:cp.ndarray = cp.array(voxel_center).reshape((3, 1))
            self.voxel_steps:cp.ndarray = cp.array(voxel_steps).reshape((3, 1))

            print('ViewPlane Message: Shapes Test')
            print(f'\t{self.voxel_start.shape = }')
            print(f'\t{self.voxel_center.shape = }')
            print(f'\t{self.voxel_steps.shape = }')

        else:
            self.default_value = np.zeros((3, self.texture_dim, self.texture_dim), dtype = np.float32)
            self.default_value[self.dim_list] = np.indices((self.texture_dim, self.texture_dim), dtype = np.float32) - self.texture_center
            self.default_value = self.default_value.reshape((3, self.texture_dim*self.texture_dim))
            self.default_value = G.QTN_DICT[G.VIEW].rotate(self.default_value, axis = 0)

        self.current_value = 1.0*self.default_value
        self.previous_value = 1.0*self.default_value
        self.difference_value = self.current_value - self.previous_value


    def __repr__(self):
        return f'{self.current_value}'
    

    def __call__(self):
        return self.current_value
    

    def set_voxel_info(self, 
                       new_voxel_start, 
                       new_voxel_center, 
                       new_voxel_steps):
        
        self.voxel_start = 1.0*new_voxel_start
        self.voxel_center = 1.0*new_voxel_center
        self.voxel_steps = 1.0*new_voxel_steps

    def set_difference_value(self):
        self.difference_value = self.current_value - self.previous_value
        

    def set_previous_value(self):
        self.previous_value = 1.0*self.current_value
        

    def set_current_value(self, 
                          new_value:np.ndarray|cp.ndarray):
        self.current_value = 1.0*new_value
        

    def update_values(self, 
                      new_current:np.ndarray|cp.ndarray, 
                      decimals = 0):
        if decimals > 0:
            new_current = new_current.round(decimals = decimals)
        self.set_previous_value()
        self.set_current_value(new_current)
        self.set_difference_value()
        

    def reset(self):
        self.set_current_value(1.0*self.default_value)
        self.set_previous_value()
        self.set_difference_value()

    
    def get_coords(self, 
                   texture_x, 
                   texture_y) -> np.ndarray|cp.ndarray:
        index = np.ravel_multi_index((texture_y, texture_x), (self.texture_dim, self.texture_dim))
        return self.current_value[:, index]
    
    def get_voxel_coords(self, 
                         texture_x, 
                         texture_y,
                         voxel_center = None,
                         voxel_start = None,
                         voxel_steps = None) -> np.ndarray|cp.ndarray:
        
        voxel_center = voxel_center if isinstance(voxel_center, cp.ndarray) else self.voxel_center
        voxel_start = voxel_start if isinstance(voxel_start, cp.ndarray) else self.voxel_start
        voxel_steps = voxel_steps if isinstance(voxel_steps, cp.ndarray) else self.voxel_steps

        index = np.ravel_multi_index((texture_y, texture_x), (self.texture_dim, self.texture_dim))
        return (self.current_value[:, index] + voxel_center - voxel_start) / voxel_steps
    
    def get_voxel_view(self, 
                       transpose = True,
                       voxel_center = None,
                       voxel_start = None,
                       voxel_steps = None) -> cp.ndarray:
        
        voxel_center = voxel_center if isinstance(voxel_center, cp.ndarray) else self.voxel_center
        voxel_start = voxel_start if isinstance(voxel_start, cp.ndarray) else self.voxel_start
        voxel_steps = voxel_steps if isinstance(voxel_steps, cp.ndarray) else self.voxel_steps

        if transpose:
            return (self.current_value + voxel_center - voxel_start) / voxel_steps
        else:
            return ((self.current_value + voxel_center - voxel_start) / voxel_steps).T

    def set_info(self, info_dict: dict):
        self.__dict__ = dict(info_dict)

    def get_info(self) -> dict:
        return dict(self.__dict__)

    def _cleanup_(self):
        attrib_list = list(self.__dict__.keys())
        while len(attrib_list) > 0:
            attrib_key = attrib_list.pop()
            setattr(self, attrib_key, None)
            delattr(self, attrib_key)


def qtn_rotate(quaternion:qtn.QuaternionicArray, 
               vector:VectorValue|ViewPlane|ArrayValue, 
               axis = -1):
    if G.GPU_MODE:
        m = cp.array(quaternion.to_rotation_matrix)
        tensordot_axis = m.ndim - 2
        final_axis = tensordot_axis + (axis % vector.ndim)
        return cp.moveaxis(
                cp.tensordot(
                    m, 
                    vector, 
                    axes = (-1, axis)), 
                tensordot_axis, 
                final_axis)

    else:
        return quaternion.rotate(vector, axis = axis)
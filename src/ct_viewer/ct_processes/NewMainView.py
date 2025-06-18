from .Globals import *
from . import VolumeLayer

class MainView:
    mode_return_type = lambda x, y: x.astype(getattr(cp, y)) if G.GPU_MODE else lambda x, y: x.astype(getattr(np, y))
    mode_return_array = lambda x: x.get() if G.GPU_MODE else lambda x: x
    mode_create_array = lambda x: cp.array(x) if G.GPU_MODE else lambda x: np.array(x)
    mode_function = lambda x: getattr(cp, x.__name__) if G.GPU_MODE else getattr(np, x.__name__)
    mode_number = lambda x: getattr(cp, x.__str__()) if G.GPU_MODE else getattr(np, x.__str__())\
    
    def __init__(self, 
                 VolumeLayerGroups:VolumeLayer.VolumeLayerGroups):
        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()
            print(f'MainView Message: Using CUDA device {G.DEVICE}.')

        self.VolumeLayerGroups:VolumeLayer.VolumeLayerGroups = VolumeLayerGroups
        self.window_tag:str = ''
        self.drawlist_texture_tag:str = ''
        self.drawlist_colorbar_tag:str = ''
        self.colormap_texture_tag:str = ''
        self.crosshair_vertical_tag:str = ''
        self.crosshair_horizontal_tag:str = ''
        self.crosshar_drawlayer_tag:str = ''
        self.texture_drawlayer_tag:str = ''
        self.mouse_pos_draw_layer_tag:str = ''
        self.crosshair_pos_draw_layer_tag:str = ''
        self.mouse_pos_texture_info_text:str = ''
        self.crosshair_pos_texture_info_text:str = ''
        self.landmark_draw_layer_tag:str = ''
        self.window_dict = {}

    # I am writing these in this way so we can have multiple draw windows at some point. 
    # These should really be classes of their own. 
    def return_texture_drawlist_tag(self, window_tag:str = '') -> str:
        if window_tag == '':
            window_tag = self.window_tag
        return self.window_dict[window_tag]['TextureDrawList']
    
    def return_colormap_tag(self, window_tag:str = '') -> str:
        if window_tag == '':
            window_tag = self.window_tag
        return self.window_dict[window_tag]['TextureColormap']
    
    def return_texture_drawlayer_tag(self, window_tag:str = '') -> str:
        if window_tag == '':
            window_tag = self.window_tag
        return self.window_dict[window_tag]['TextureDrawLayer']
    
    def return_colormap_drawlayer_tag(self, window_tag:str = '') -> str:
        if window_tag == '':
            window_tag = self.window_tag
        return self.window_dict[window_tag]['ColormapDrawList']
    
    def return_mouse_pos_texture_info_text_tag(self, window_tag:str = '') -> str:
        if window_tag == '':
            window_tag = self.window_tag
        return self.window_dict[window_tag]['TextureDrawTextMousePosInfo']
    
    def return_crosshair_pos_texture_info_text_tag(self, window_tag:str = '') -> str:
        if window_tag == '':
            window_tag = self.window_tag
        return self.window_dict[window_tag]['TextureDrawTextCrosshairPosInfo']
    
    def return_landmark_drawlayer_tag(self, window_tag:str = '') -> str:
        if window_tag == '':
            window_tag = self.window_tag
        return self.window_dict[window_tag]['TextureDrawLandmarks']
    
    def create_draw_window(self, 
                           window_tag:str, 
                           parent_window:str,
                           item_handler_reg_tag:str|int = None):
        
        self.window_tag = create_tag('NewMainView', 'ChildWindow', window_tag)
        self.drawlist_texture_tag = create_tag(self.window_tag, 'DrawList', 'Texture')
        self.drawlist_colorbar_tag = create_tag(self.window_tag, 'DrawList', 'Colorbar')
        self.colormap_texture_tag = create_tag(self.window_tag, 'Colormap', 'Texture')
        self.crosshair_vertical_tag = create_tag(self.window_tag, 'Line', 'CrosshairVertical')
        self.crosshair_horizontal_tag = create_tag(self.window_tag, 'Line', 'CrosshairHorizontal')
        self.chrosshair_draw_layer_tag = create_tag(self.window_tag, 'DrawLayer', 'CrosshairLayer')
        self.texture_draw_layer_tag = create_tag(self.window_tag, 'DrawLayer', 'TextureLayer')
        self.mouse_pos_draw_layer_tag = create_tag(self.window_tag, 'DrawLayer', 'MousePosInfo')
        self.crosshair_pos_draw_layer_tag = create_tag(self.window_tag, 'DrawLayer', 'CrosshairPosInfo')
        self.mouse_pos_texture_info_text = create_tag(self.window_tag, 'DrawText', 'MousePosInfo')
        self.crosshair_pos_texture_info_text = create_tag(self.window_tag, 'DrawText', 'CrosshairPosInfo')
        self.landmark_draw_layer_tag = create_tag(self.window_tag, 'DrawLayer', 'Landmarks')
        

        self.window_dict[self.window_tag] = {'TextureDrawList': self.drawlist_texture_tag,
                                             'TextureColormap': self.colormap_texture_tag,
                                             'ColormapDrawList': self.drawlist_colorbar_tag,
                                             'TextureDrawLayer': self.texture_draw_layer_tag,
                                             'TextureCrosshairVertical': self.crosshair_vertical_tag,
                                             'TextureCrosshairHorizontal': self.crosshair_horizontal_tag,
                                             'TextureDrawLayerCrosshair': self.chrosshair_draw_layer_tag,
                                             'TextureDrawLayerMousePosText': self.mouse_pos_draw_layer_tag,
                                             'TextureDrawLayerCrosshairPosText': self.crosshair_pos_draw_layer_tag,
                                             'TextureDrawTextMousePosInfo': self.mouse_pos_texture_info_text,
                                             'TextureDrawTextCrosshairPosInfo': self.crosshair_pos_texture_info_text,
                                             'TextureDrawLandmarks': self.landmark_draw_layer_tag}
        
        
        with dpg.group(horizontal = True, 
                       parent = parent_window):
            with dpg.child_window(tag = self.window_tag, 
                                  width = G.CONFIG_DICT['app_settings']['main_pane_width'],
                                  height = G.CONFIG_DICT['app_settings']['main_pane_height']):
                with dpg.group(horizontal = True):
                    dpg.add_drawlist(width = G.CONFIG_DICT['app_settings']['main_texture_width'], 
                                     height = G.CONFIG_DICT['app_settings']['main_texture_height'],
                                     tag = self.drawlist_texture_tag)
                    
                    dpg.add_colormap_scale(min_scale=0, 
                                           max_scale=1200, 
                                           width=G.CONFIG_DICT['app_settings']['colormap_scale_width'], 
                                           height=-1, 
                                           tag = self.colormap_texture_tag, 
                                           colormap=G.DEFAULT_IMAGE_SETTINGS['colormap_scale'])
                    
        dpg.add_draw_layer(parent = self.drawlist_texture_tag, 
                           tag = self.texture_draw_layer_tag)
        
        with dpg.draw_layer(parent = self.drawlist_texture_tag,
                            tag = self.chrosshair_draw_layer_tag):

            dpg.draw_line([G.TEXTURE_CENTER, 0], 
                          [G.TEXTURE_CENTER, 1000], 
                          color = dpg.get_value(f'ValueRegister_Configuration_default_crosshair_color_value'),
                          tag = self.crosshair_vertical_tag)
            dpg.bind_item_theme(dpg.last_item(), G.LINE_THEME)
            
            dpg.draw_line([0, G.TEXTURE_CENTER], 
                          [1000, G.TEXTURE_CENTER], 
                          color = dpg.get_value(f'ValueRegister_Configuration_default_crosshair_color_value'),
                          tag = self.crosshair_horizontal_tag)
            dpg.bind_item_theme(dpg.last_item(), G.LINE_THEME)            
            
        dpg.add_draw_layer(parent = self.drawlist_texture_tag,
                           tag = self.landmark_draw_layer_tag)
        
        text_box_width = 420
        text_box_height = 95
        text_box_alpha = 175

        mouse_text_box_start = [0, 0]
        mouse_text_box_end = [mouse_text_box_start[0] + text_box_width,
                              mouse_text_box_start[1] + text_box_height]
        mouse_text_start = [mouse_text_box_start[0] + 5,
                            mouse_text_box_start[1] + 5]
        initial_mouse_text = \
"""                    (X    , Y    , Z    , HU   )
Texture Position  : (0.000, 0.000)
Physical Position : (0.000, 0.000, 0.000)
Voxel Position    : (0.000, 0.000, 0.000)
Mouse Position    : (0.000, 0.000, 0.000, 0.000)"""
        with dpg.draw_layer(parent = self.drawlist_texture_tag,
                            tag = self.mouse_pos_draw_layer_tag,
                            user_data = initial_mouse_text):

            dpg.draw_rectangle(mouse_text_box_start, 
                               mouse_text_box_end, 
                               color = (0, 0, 0, text_box_alpha), 
                               fill = (0, 0, 0, text_box_alpha))
            dpg.draw_text(mouse_text_start, 
                          initial_mouse_text, 
                          user_data = mouse_text_start,
                          tag = self.mouse_pos_texture_info_text, 
                          size = 14)
            
            print('MainView Message: mouse_pos_texture_info_text position:')
            print(f'\t{dpg.get_item_pos(self.mouse_pos_texture_info_text) = }')
        
        crosshair_text_box_start = [G.TEXTURE_DIM - text_box_width, 0]
        crosshair_text_box_end = [crosshair_text_box_start[0] + text_box_width,
                                  crosshair_text_box_start[1] + text_box_height]
        crosshair_text_start = [crosshair_text_box_start[0] + 5,
                                crosshair_text_box_start[1] + 5]
        initial_text_crosshair = \
"""                    (X    , Y    , Z    , HU   )
Texture Position  : (0.000, 0.000)
Physical Position : (0.000, 0.000, 0.000)
Voxel Position    : (0.000, 0.000, 0.000)
Crosshair Position: (0.000, 0.000, 0.000, 0.000)"""
        with dpg.draw_layer(parent = self.drawlist_texture_tag,
                            tag = self.crosshair_pos_draw_layer_tag,
                            user_data = initial_text_crosshair):

            dpg.draw_rectangle(crosshair_text_box_start, 
                               crosshair_text_box_end, 
                               color = (0, 0, 0, text_box_alpha), 
                               fill = (0, 0, 0, text_box_alpha))
            dpg.draw_text(crosshair_text_start, 
                          initial_text_crosshair, 
                          user_data = crosshair_text_start,
                          tag = self.crosshair_pos_texture_info_text, 
                          size = 14)

            print('MainView Message: crosshair_pos_texture_info_text position:')
            print(f'\t{dpg.get_item_pos(self.crosshair_pos_texture_info_text) = }')

    def add_volume_to_drawlist(self, 
                               volume_layer: VolumeLayer.VolumeLayer,
                               drawlist:str = ''):
        if drawlist == '':
            drawlist = self.texture_draw_layer_tag
        
        print(f'MainView Message: Drawing {volume_layer.texture.texture_tag} on {drawlist}')

        volume_layer.add_texture_to_drawlist(drawlist = drawlist,
                                             pixel_start = [0, 0],
                                             pixel_end = [volume_layer.texture_dim, 
                                                          volume_layer.texture_dim])
        
    def _cleanup_(self):
        for w_dict in self.window_dict.values():
            for value in w_dict.values():
                print(f'MainView Message: Deleting {value}.')
                if dpg.does_item_exist(value):
                    dpg.delete_item(value)
                if dpg.does_alias_exist(value):
                    dpg.delete_item(value)

        dict_keys = list(self.__dict__.keys())
        while len(dict_keys) > 0:
            attrib_key = dict_keys.pop()
            setattr(self, attrib_key, None)
            delattr(self, attrib_key)
            cp._default_memory_pool.free_all_blocks()
from .Globals import *

class MainView:
    mode_return_type = lambda x, y: x.astype(getattr(cp, y)) if G.GPU_MODE else lambda x, y: x.astype(getattr(np, y))
    mode_return_array = lambda x: x.get() if G.GPU_MODE else lambda x: x
    mode_create_array = lambda x: cp.array(x) if G.GPU_MODE else lambda x: np.array(x)
    mode_function = lambda x: getattr(cp, x.__name__) if G.GPU_MODE else getattr(np, x.__name__)
    mode_number = lambda x: getattr(cp, x.__str__()) if G.GPU_MODE else getattr(np, x.__str__())\
    
    def __init__(self):
        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()
            print(f'MainView Message: Using CUDA device {G.DEVICE}.')

        self.window_dict:dict = {}


    def get_window_tags(self):
        return list(self.window_dict.keys())


    def create_window_dict(self, window_tag:str, height:int, width: int):

        window_dict = {'TextureDrawList': create_tag(window_tag, 'DrawList', 'Texture'),
                       'TextureColormap': create_tag(window_tag, 'Colormap', 'Texture'),
                       'ColormapDrawList': create_tag(window_tag, 'DrawList', 'Colorbar'),
                       'TextureDrawLayer': create_tag(window_tag, 'DrawLayer', 'TextureLayer'),
                       'TextureCrosshairVertical': create_tag(window_tag, 'Line', 'CrosshairVertical'),
                       'TextureCrosshairHorizontal': create_tag(window_tag, 'Line', 'CrosshairHorizontal'),
                       'TextureDrawLayerCrosshair': create_tag(window_tag, 'DrawLayer', 'CrosshairLayer'),
                       'TextureDrawLayerMousePosText': create_tag(window_tag, 'DrawLayer', 'MousePosInfo'),
                       'TextureDrawLayerCrosshairPosText': create_tag(window_tag, 'DrawLayer', 'CrosshairPosInfo'),
                       'TextureDrawTextMousePosInfo': create_tag(window_tag, 'DrawText', 'MousePosInfo'),
                       'TextureDrawTextCrosshairPosInfo': create_tag(window_tag, 'DrawText', 'CrosshairPosInfo'),
                       'TextureDrawLandmarks': create_tag(window_tag, 'DrawLayer', 'Landmarks'),
                       'height': height,
                       'width': width}
        
        return window_dict
    
    def get_landmark_drawlist_tags(self, window_tags:list[str] = [None]) -> list[str]:
        return [self.window_dict[tag]['TextureDrawLandmarks'] for tag in self.get_window_tags()]
    
    def get_texture_drawlist_tags(self, window_tags:list[str] = [None]) -> list[str]:
        return [self.window_dict[tag]['TextureDrawList'] for tag in self.get_window_tags()]

    # I am writing these in this way so we can have multiple draw windows at some point. 
    # These should really be classes of their own. 
    def return_texture_drawlist_tag(self, window_tag:str = '') -> str:
        if window_tag == '':
            window_tag = self.get_window_tags()[0]
        return self.window_dict[window_tag]['TextureDrawList']
    
    
    def return_colormap_tag(self, window_tag:str = '') -> str:
        if window_tag == '':
            window_tag = self.get_window_tags()[0]
        return self.window_dict[window_tag]['TextureColormap']


    def return_colormap_drawlayer_tag(self, window_tag:str = '') -> str:
        if window_tag == '':
            window_tag = self.get_window_tags()[0]
        return self.window_dict[window_tag]['ColormapDrawList']


    def return_texture_drawlayer_tag(self, window_tag:str = '') -> str:
        if window_tag == '':
            window_tag = self.get_window_tags()[0]
        return self.window_dict[window_tag]['TextureDrawLayer']


    def return_crosshair_vertical_tag(self, window_tag:str = '') -> str:
        if window_tag == '':
            window_tag = self.get_window_tags()[0]
        return self.window_dict[window_tag]['TextureCrosshairVertical']
    

    def return_crosshair_horizontal_tag(self, window_tag:str = '') -> str:
        if window_tag == '':
            window_tag = self.get_window_tags()[0]
        return self.window_dict[window_tag]['TextureCrosshairHorizontal']
    

    def return_crosshair_drawlayer_tag(self, window_tag:str = '') -> str:
        if window_tag == '':
            window_tag = self.get_window_tags()[0]
        return self.window_dict[window_tag]['TextureDrawLayerCrosshair']


    def return_mouse_pos_drawlayer_tag(self, window_tag:str = '') -> str:
        if window_tag == '':
            window_tag = self.get_window_tags()[0]
        return self.window_dict[window_tag]['TextureDrawLayerMousePosText']
    

    def return_crosshair_pos_drawlayer_tag(self, window_tag:str = '') -> str:
        if window_tag == '':
            window_tag = self.get_window_tags()[0]
        return self.window_dict[window_tag]['TextureDrawLayerCrosshairPosText']


    def return_mouse_pos_texture_info_text_tag(self, window_tag:str = '') -> str:
        if window_tag == '':
            window_tag = self.get_window_tags()[0]
        return self.window_dict[window_tag]['TextureDrawTextMousePosInfo']
    

    def return_crosshair_pos_texture_info_text_tag(self, window_tag:str = '') -> str:
        if window_tag == '':
            window_tag = self.get_window_tags()[0]
        return self.window_dict[window_tag]['TextureDrawTextCrosshairPosInfo']
    

    def return_landmark_drawlayer_tag(self, window_tag:str = '') -> str:
        if window_tag == '':
            window_tag = self.get_window_tags()[0]
        return self.window_dict[window_tag]['TextureDrawLandmarks']


    def create_inset_window(self, 
                            window_tag:str,
                            parent_window:str,
                            start:list[int, int] = [0, 0],
                            stop:list[int, int] = [-1, -1],
                            width:int = -1,
                            height:int = -1,
                            item_handler_reg_tag:str|int = None):
        
        window_tag = create_tag('NewMainView', 'ChildWindow', window_tag)
        self.window_dict[window_tag] = self.create_window_dict(window_tag, height, width)
        
        with dpg.child_window(parent = parent_window,
                              tag = window_tag,
                              user_data = self.window_dict[window_tag],
                              border = False, 
                              no_scrollbar = True, 
                              pos = start,
                              width = width,
                              height = height,
                              frame_style = False):
            
            dpg.add_drawlist(width = width, 
                             height = height,
                             tag = self.return_texture_drawlist_tag(window_tag = window_tag))
            
            dpg.add_draw_layer(parent = self.return_texture_drawlist_tag(window_tag = window_tag), 
                               tag = self.return_texture_drawlayer_tag(window_tag = window_tag),
                               user_data = {'start': start, 
                                            'stop': stop,
                                            'center_loc': [round(width / 2), 
                                                           round(height / 2)],
                                            'width': width,
                                            'height': height})
            
            with dpg.draw_layer(parent = self.return_texture_drawlist_tag(window_tag = window_tag),
                                tag = self.return_crosshair_drawlayer_tag(window_tag = window_tag)):

                dpg.draw_line([round(0.5 * width), 0.0], 
                              [round(0.5 * width), height], 
                              color = dpg.get_value(f'ValueRegister_Configuration_default_crosshair_color_value'),
                              tag = self.return_crosshair_vertical_tag(window_tag = window_tag))
                dpg.bind_item_theme(dpg.last_item(), G.LINE_THEME)
                
                dpg.draw_line([0.0, round(0.5 * height)], 
                              [width, round(0.5 * height)], 
                              color = dpg.get_value(f'ValueRegister_Configuration_default_crosshair_color_value'),
                              tag = self.return_crosshair_horizontal_tag(window_tag = window_tag))
                dpg.bind_item_theme(dpg.last_item(), G.LINE_THEME)
                
            dpg.add_draw_layer(parent = self.return_texture_drawlist_tag(window_tag = window_tag),
                               tag = self.return_landmark_drawlayer_tag(window_tag = window_tag))
        
    def create_draw_window(self, 
                           window_tag:str, 
                           parent_window:str,
                           item_handler_reg_tag:str|int = None):
        
        window_tag = create_tag('NewMainView', 'ChildWindow', window_tag)
        self.window_dict[window_tag] = self.create_window_dict(window_tag, 
                                                               G.CONFIG_DICT['app_settings']['main_texture_height'],
                                                               G.CONFIG_DICT['app_settings']['main_texture_width'])

        
        with dpg.group(horizontal = True, 
                       parent = parent_window):
            with dpg.child_window(tag = window_tag, 
                                  width = G.CONFIG_DICT['app_settings']['main_pane_width'],
                                  height = G.CONFIG_DICT['app_settings']['main_pane_height'],
                                  border = False):
                with dpg.group(horizontal = True):
                    dpg.add_drawlist(width = G.CONFIG_DICT['app_settings']['main_texture_width'], 
                                     height = G.CONFIG_DICT['app_settings']['main_texture_height'],
                                     tag = self.return_texture_drawlist_tag(window_tag))
                    
                    dpg.add_colormap_scale(min_scale=0, 
                                           max_scale=1200, 
                                           width=G.CONFIG_DICT['app_settings']['colormap_scale_width'], 
                                           height=G.TEXTURE_DIM, 
                                           tag = self.return_colormap_tag(window_tag),
                                           colormap=G.DEFAULT_IMAGE_SETTINGS['colormap_scale'])
        
        dpg.add_draw_layer(parent = self.return_texture_drawlist_tag(window_tag), 
                           tag = self.return_texture_drawlayer_tag(window_tag), 
                           user_data = {'start': [0, 0], 
                                        'stop': [G.TEXTURE_DIM, G.TEXTURE_DIM],
                                        'center_loc': [G.TEXTURE_CENTER, 
                                                       G.TEXTURE_CENTER],
                                        'width': G.TEXTURE_DIM,
                                        'height': G.TEXTURE_DIM})
        
        with dpg.draw_layer(parent = self.return_texture_drawlist_tag(window_tag),
                            tag = self.return_crosshair_drawlayer_tag(window_tag)):

            dpg.draw_line([G.TEXTURE_CENTER, 0], 
                          [G.TEXTURE_CENTER, 1000], 
                          color = dpg.get_value(f'ValueRegister_Configuration_default_crosshair_color_value'),
                          tag = self.return_crosshair_vertical_tag(window_tag))
            dpg.bind_item_theme(dpg.last_item(), G.LINE_THEME)
            
            dpg.draw_line([0, G.TEXTURE_CENTER], 
                          [1000, G.TEXTURE_CENTER], 
                          color = dpg.get_value(f'ValueRegister_Configuration_default_crosshair_color_value'),
                          tag = self.return_crosshair_horizontal_tag(window_tag))
            dpg.bind_item_theme(dpg.last_item(), G.LINE_THEME)            
            
        dpg.add_draw_layer(parent = self.return_texture_drawlist_tag(window_tag),
                           tag = self.return_landmark_drawlayer_tag(window_tag))
        
        
        # TEXT BOX
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
        with dpg.draw_layer(parent = self.return_texture_drawlist_tag(window_tag),
                            tag = self.return_mouse_pos_drawlayer_tag(window_tag),
                            user_data = initial_mouse_text):

            dpg.draw_rectangle(mouse_text_box_start, 
                               mouse_text_box_end, 
                               color = (0, 0, 0, text_box_alpha), 
                               fill = (0, 0, 0, text_box_alpha))
            dpg.draw_text(mouse_text_start, 
                          initial_mouse_text, 
                          user_data = mouse_text_start,
                          tag = self.return_mouse_pos_texture_info_text_tag(window_tag),
                          size = 14)
        
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
        with dpg.draw_layer(parent = self.return_texture_drawlist_tag(window_tag),
                            tag = self.return_crosshair_pos_drawlayer_tag(window_tag),
                            user_data = initial_text_crosshair):

            dpg.draw_rectangle(crosshair_text_box_start, 
                               crosshair_text_box_end, 
                               color = (0, 0, 0, text_box_alpha), 
                               fill = (0, 0, 0, text_box_alpha))
            dpg.draw_text(crosshair_text_start, 
                          initial_text_crosshair, 
                          user_data = crosshair_text_start,
                          tag = self.return_crosshair_pos_texture_info_text_tag(window_tag),
                          size = 14)
        
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
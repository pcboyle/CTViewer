from .Globals import *

class Themes():
    def __init__(self):
        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()
            
        with dpg.theme(tag = G.LINE_THEME):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvPlotCol_Line, 
                                    G.CONFIG_DICT['default_colors']['default_crosshair_color'], 
                                    tag = 'crosshair_color_theme', 
                                    category = dpg.mvThemeCat_Plots)

                # dpg.add_theme_color(dpg.mvPlotCol_Line, 
                #                     G.CONFIG_DICT['default_colors']['default_crosshair_color'], 
                #                     tag = 'crosshair_box_color_theme', 
                #                     category = dpg.mvThemeCat_Plots)
                
        self.colormap_registries = []

    def register_colormaps(self, 
                           colormap_dict: dict,
                           colormap_registry: str):
        
        if colormap_registry not in self.colormap_registries:
            self.colormap_registries.append(colormap_registry)

        for cmap_key in colormap_dict.keys():
            
            cmap_name = f'{G.COLORMAP_DICT[cmap_key]["name"]}' # eg, m_fire
            cmap_name_r = f'{cmap_name}_r'

            rgb_values = getattr(G.COLORMAP_DICT[cmap_key]['module'], cmap_name)(list(range(256)))
            rgb_values_r = getattr(G.COLORMAP_DICT[cmap_key]['module'], cmap_name_r)(list(range(256)))

            dpg.add_colormap(list(np.round(rgb_values*255.0, decimals = 4)), 
                             False, 
                             parent = colormap_registry, 
                             tag=f'colormap_{cmap_name}')
            
            dpg.add_colormap(list(np.round(rgb_values_r*255.0, decimals = 4)), 
                             False, 
                             parent = colormap_registry, 
                             tag=f'colormap_{cmap_name_r}')

            print(f'Themes Message: {cmap_key} registered')


    def colorize_texture(self, 
                         colormap_name: str,
                         texture: np.ndarray) -> np.ndarray:
        return dpg.sample_colormap(f'colormap_{colormap_name}', texture.flatten().tolist())
    

    def _cleanup_(self):
        dict_keys = list(self.__dict__.keys())
        while len(dict_keys) > 0:
            attrib_key = dict_keys.pop()
            setattr(self, attrib_key, None)
            delattr(self, attrib_key)
            cp._default_memory_pool.free_all_blocks()
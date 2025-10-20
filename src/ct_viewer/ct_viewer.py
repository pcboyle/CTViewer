import sys
import argparse
import datetime
import time
from pathlib import Path
import platform

try:
    from .ct_processes.Globals import *
    
except:
    from ct_processes.Globals import *

try: 
    from .__init__ import __version__

except:
    from __init__ import __version__


_MAXFRAMERATE_ = 1/80

def get_frame_rate(s_time,
                   frame_increment):
    frame_datetime = datetime.datetime.now()
    frame_delta:datetime.timedelta = (frame_datetime - s_time)
    frames_per_second = round(frame_increment / (frame_delta.seconds + frame_delta.microseconds / 1e6))
    if frames_per_second > 500:
        frames_per_second = 0
    return frames_per_second


def get_gpu_memory(mempool_bytes, 
                   f_count, 
                   f_space,
                   f_rate):
    
    frame_datetime = datetime.datetime.now()
    hour = f"{frame_datetime.hour}".zfill(2)
    minute = f"{frame_datetime.minute}".zfill(2)
    second = f"{frame_datetime.second}".zfill(2)
    microsecond = f"{frame_datetime.microsecond}".zfill(6)
    frame_time_string = f'{hour}:{minute}:{second}.{microsecond}    '
    mempool_used_bytes = round(float(mempool_bytes)/1e6, ndigits=3)
    gpu_string = f'{frame_time_string}FRAME: {f_count:>10}'
    gpu_string = f'{gpu_string}{f_space}FPS: {f_rate:>3d}'
    gpu_string = f'{gpu_string}{f_space}MemPool Bytes       : {mempool_used_bytes:>10}\n'
    return gpu_string


def get_gpu_utilization():
    pass


def get_gpu_information():
    gpu_device_info = {}
    try:
        for device_n in range(cp.cuda.runtime.getDeviceCount()):
            device_properties = cp.cuda.runtime.getDeviceProperties(device_n)
            gpu_device_info[f'{device_n}'] = {'name': device_properties['name'].decode(),
                                              'totalGlobalMem': device_properties['totalGlobalMem'] / 1.074e9}
    except:
        pass

def main():

    parser = argparse.ArgumentParser(
        prog='CT Viewer',
        description = 'A tool to examine multiple ct volumes.',
    )

    parser.add_argument('-debug', '--debug', help = 'Turn debug mode on.', action = 'store_true')
    parser.add_argument('-gpu', '--gpu', nargs='?', type=int, action='store', const = 0, default = 0,
                        help='Enable GPU computation and select device number. Default device is 0.')

    args = parser.parse_args()
    print(f'APP Message: arguments parsed: {args = }')

    cpu_name = f'{platform.processor()}'
    gpu_name = ''
    gpu_mempool = None
    gpu_pinned_mempool = None


    if args.gpu > -1:
        if args.gpu >= cp.cuda.runtime.getDeviceCount():
            print(f'APP Message: Device {args.gpu = } not valid. Max device index is {cp.cuda.runtime.getDeviceCount() - 1}.')
            print(f'Running using GPU device {0}: ')
            
        else:
            setattr(G, 'GPU_MODE', True)
            setattr(G, 'DEVICE', 1*args.gpu)
            cp.cuda.Device(args.gpu).use()

            gpu_name = cp.cuda.runtime.getDeviceProperties(args.gpu)["name"].decode()

            gpu_mempool = cp.get_default_memory_pool()
            gpu_pinned_mempool = cp.get_default_pinned_memory_pool()

    else:
        setattr(G, 'GPU_MODE', False)

    if args.debug:
        setattr(G, 'DEBUG_MODE', True)

    else:
        setattr(G, 'DEBUG_MODE', False)

    # Get date and time of instance using shell commands. 
    month_dict = {1: 'JAN',
                  2: 'FEB',
                  3: 'MAR',
                  4: 'APR',
                  5: 'MAY', 
                  6: 'JUN',
                  7: 'JUL',
                  8: 'AUG', 
                  9: 'SEP',
                  10: 'OCT', 
                  11: 'NOV',
                  12: 'DEC'}
    
    app_datetime:datetime.datetime = datetime.datetime.now()
    day = f'{app_datetime.day}'.zfill(2)
    month = month_dict[app_datetime.month]
    year = app_datetime.year
    hour = f'{app_datetime.hour}'.zfill(2)
    minute = f'{app_datetime.minute}'.zfill(2)
    second = f'{app_datetime.second}'.zfill(2)
    microsecond = f'{app_datetime.microsecond}'.zfill(6)
    app_datetime_string = f'{day}{month}{year}-{hour}{minute}{second}.{microsecond}'

    gpu_log_path:Path = Path(G.LOG_DIR).joinpath(f'GPU_{app_datetime_string}.LOG')
    gpu_log_path.touch()
    initial_log_text = f'GPU NAME: {gpu_name}\nCPU NAME: {cpu_name}'
    initial_log_text = f'{initial_log_text}\nDATE-TIME: {app_datetime_string}'
    initial_log_text = f'{initial_log_text}\n{"-"*50}\n'
    initial_log_text = f'{initial_log_text}'
    gpu_log_path.write_text(initial_log_text)

    G.initialize_colormaps()

    dpg.create_context()
    
    try:
        from .ct_processes import CTViewer as ctv

    except: 
        from ct_processes import CTViewer as ctv

    ct_viewer = ctv(dpg.window(label = "", 
                               tag = G.ROOT,
                               width = G.CONFIG_DICT['app_settings']['app_width'], 
                               height = G.CONFIG_DICT['app_settings']['app_height']))

    if G.GPU_MODE:
        cp.cuda.Device(G.DEVICE).use()
        # dpg.configure_app(auto_device=False, device = G.DEVICE, wait_for_input=False, manual_callback_management=True)
        dpg.configure_app(auto_device=False, device = G.DEVICE, wait_for_input=False)
        gpu_debug_file = Path(G.CONFIG_DIR).joinpath('gpu_debug')

    else:
        dpg.configure_app(auto_device=True, wait_for_input=False)

    if G.GPU_MODE:
        window_title = f'CT Viewer {__version__}, GPU {gpu_name}' 
    else:
        window_title = f'CT Viewer {__version__}'

    dpg.create_viewport(title = window_title, 
                        decorated = True,
                        width = G.CONFIG_DICT['app_settings']['app_width'], 
                        height = G.CONFIG_DICT['app_settings']['app_height'], 
                        x_pos = 0, 
                        y_pos = 0,
                        vsync = False)
    
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window(G.ROOT, True)
    dpg.focus_item(G.ROOT)
    
    if G.DEBUG_MODE:
        dpg.show_style_editor()
        dpg.show_item_registry()
        dpg.show_tool(dpg.mvTool_Metrics)
        dpg.show_debug()
        try:
            while dpg.is_dearpygui_running():
            # insert here any code you would like to run in the render loop
            # you can manually stop by using stop_dearpygui()
                if G.FILE_LOADED:
                    for debug_tag in G.DEBUG_INFO_TAGS:
                        dpg.set_value(f'{debug_tag}_debug_info', f'{getattr(G.APP.main_view, debug_tag)}')
                
                dpg.render_dearpygui_frame()
                
        finally:
            G.save_config('current')
            ct_viewer._cleanup_()
            dpg.destroy_context()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
                
        return 0
    
    else:
        frame_count:int = 0
        frame_space = " "*10
        start_time = datetime.datetime.now()
        frame_rate = 0
        try:
            gpu_log = open(gpu_log_path, mode = 'a')
            while dpg.is_dearpygui_running():
                # jobs = dpg.get_callback_queue() # retrieves and clears queue
                # if jobs:
                #     print(jobs)
                # dpg.run_callbacks(jobs)
                dpg.render_dearpygui_frame()
                if frame_count%60 == 0:

                    frame_rate = get_frame_rate(start_time, 60)
                    print(get_gpu_memory(gpu_mempool.used_bytes(), 
                                            frame_count, 
                                            frame_space, 
                                            frame_rate), 
                            file = gpu_log, 
                            flush = True, 
                            end = '')
                    
                    start_time = datetime.datetime.now()                
                frame_count += 1
                time.sleep(_MAXFRAMERATE_)

        except:
            with Exception as e:
                print(f'{e}')
        
        finally:
            print(get_gpu_memory(gpu_mempool.used_bytes(), frame_count, frame_space, frame_rate), 
                            file = gpu_log, 
                            flush = True, 
                            end = '')
            G.save_config('current')
            ct_viewer._cleanup_()
            dpg.destroy_context()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            print(get_gpu_memory(gpu_mempool.used_bytes(), frame_count + 1, frame_space, frame_rate), 
                            file = gpu_log, 
                            flush = True, 
                            end = '')
            print('-'*50, 
                    file = gpu_log, 
                    flush = True, 
                    end = '')
            
            gpu_log.close()
    
        return 0
    
if __name__ == '__main__':
    err = main()
    if err:
        sys.exit(err)
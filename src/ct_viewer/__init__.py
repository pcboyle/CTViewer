__version__ = "0.3.3.2.2"

# TODO:
#       Zoom Feature                        --  Done. 
#       Operations Feature                  --  Initial operations complete. May require writing custom GPU kernels. 
#       Histogram Feature                   --  Added. Fixed switching between Layer and Group.  
#       Mouse and Crosshair Readouts        --  Actually Done. Uses VolumeLayerGroups, primarily. Went on wild goose chase for cause of double motion errors. It was in the texture itself with x_shift, y_shift. 
#           - Camera vs Volume Control      --  
#       Landmarking Feature                 --  Added Remove and Save. Need to ensure that Load works as expected. 
#       Configuration Feature               --  Need to connect the configuration menu with the values. 
#       Keyboard + Mouse Controls           --  Done. KBM controls added to OptionsPanel. Need to add zoom and angle controls to the mouse. 
#       GPU Log                             --  Initial version done. Saves running total GPU memory utilization in .ct_viewer folder. 
#       Layer Permanency                    --  Done. Each VolumeLayer now contains it's own independent orientation and intensity information. 
#       FileDialog Change Drive             --  Fix flashing as you change the drive. Should be done and rendered on the same frame. May have to do with modal window redrawing it. 
#       VolumeLayerGroup Orientation        --  Fix center locations used to get the physical voxel array positions. 

# TODO: ADVANCED LAYER MANIPULATION
#       Overlay Layers                      --  I want to add the ability to overlay and control the transparency of each layer. 
#       Drawing/Querying Layers             --  I want to be able to overlay different volume layers to see how they match up. 
Command line options: 
    To start program, use ct_viewer. 

    Can enter debug mode with -debug

    Select GPU with -gpu=<device number>
        Default is -gpu=0

Config file:

    A configuration file will now be created/loaded at $USERHOME/.config/ct_viewer/config.ini


Geometry:
    All files are loaded in and interpolated to be 1 x 1 x 1 mm/pix using the pixel dimensions, slice thickness, and volume affine. 
    They are initially centered at (0, 0, 0) in physical space in mm. 

    If no affine is present, assumes a 1x1x1 mm^3 voxel dimension and positively aligned volume. 

Navigation: 

    All navigation is done from the perspective of the patient according to the affine
    of the volume. 

            a       -   Move volume in the negative X direction   -   Patient moves to their Right
            d       -   Move volume in the positive X direction   -   Patient moves to their Left

            w       -   Move volume in the negative Y direction   -   Patient moves to their Anterior
            s       -   Move volume in the positive Y direction   -   Patient moves to their Posterior

            q       -   Move volume in the negative Z direction   -   Patient moves to their Inferior
            e       -   Move volume in the positive Z direction   -   Patient moves to their Superior

            z       -   Negative change in image index            -
            c       -   Positive change in image index            -

    Shift + a       -   Negative pitch of volume                  -   Rotates about X, moves Z into Y. 
    Shift + d       -   Positive pitch of volume                  -   Rotates about X, moves Y into Z. 

    Shift + w       -   Negative yaw of volume                    -   Rotates about Y, moves Z into X. 
    Shift + s       -   Positive yaw of volume                    -   Rotates about Y, moves X into Z. 

    Shift + q       -   Negative roll of volume                   -   Rotates about Z, moves Y into X. 
    Shift + e       -   Positive roll of volume                   -   Rotates about Z, moves X into Y. 

    Shift + z       -   Swaps between Group and Local control     -  
    Shift + c       -   Swaps between Group and Local control     - 

    Alt + a         -   Negative zoom in X direction              -   
    Alt + d         -   Positive zoom in X direction              -   

    Alt + w         -   Negative zoom in Y direction              -   
    Alt + s         -   Positive zoom in Y direction              -   
    
    Alt + q         -   Negative zoom in Z direction              -   
    Alt + e         -   Positive zoom in Z direction              -   

    Crtl + Any      -   Changes navigation setting at 5x the current increment. 
                    -   Does not effect z and c keys. 

    Middle Mouse    -   Click and drag to move the volume. Currently a little floaty as the rendering and mouse motion aren't perfectly aligned. 

    Spacebar        -   Create landmark at crosshair position


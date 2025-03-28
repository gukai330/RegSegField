from typing import Any, Callable, List
from viser import (
    ViserServer,
)

from nerfstudio.viewer.viewer_elements import (
    ViewerDropdown,
)

from torch import nn


import numpy as np


class MeshVisualizer(ViewerDropdown):
    
    def __init__(self, 
                 meshes: List[Any] = [],
                 ):

          
        name = "mesh"
        default_value = "0"
        options = [str(i) for i in range(len(meshes[0]))]

        options += ["all", "none"]

        disabled = False
        visible = True
        # cb_hook = self.show_meshes



        super().__init__(name, default_value, options, disabled, visible, )

        self.meshes = meshes

        self.meshes_handles_mask = {}
        


    def _create_gui_handle(self, viser_server: ViserServer):

        super()._create_gui_handle(viser_server)

        self.viser_server = viser_server
        
        for o in self.options:
            
            if o == "all" or o == "none":
                continue

            handels_view = []

            color = np.random.rand(3)

            for v,mesh in enumerate(self.meshes,):
                
                try:
                    single_mesh = mesh[int(o)]
                except:
                    continue
                
                handle = viser_server.add_mesh_simple(
                    name = f"mesh_{v}_{o}",
                    vertices = single_mesh[0]*10.0,
                    faces = single_mesh[1],
                    color=color,
                    opacity=0.3,
                    visible=False,
                    side="double",
                )
                handels_view.append(handle)
            self.meshes_handles_mask.update({o:handels_view})

        self.gui_handle.on_update(lambda _: self.show_meshes())



    def show_meshes(self):
        with self.viser_server.atomic():

            if self.value == "all":
                for k, v in self.meshes_handles_mask.items():
                    if int(k) <= 3:
                        continue
                    for vv in v:
                        vv.visible = True
            elif self.value == "none":
                for v in self.meshes_handles_mask.values():
                    for vv in v:
                        vv.visible = False

            else:
                for o in self.options:
                    for v in self.meshes_handles_mask[o]:
                        if o == self.value:
                            v.visible = True
                        else:
                            v.visible = False






    


    
    

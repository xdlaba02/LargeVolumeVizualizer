import os
import conan

class LargeVolumeVizualizerRecipe(conan.ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps"

    def requirements(self):
        self.requires("opengl/system")
        self.requires("glm/0.9.9.8")
        self.requires("vc/1.4.2")
        self.requires("glfw/3.3.8")
        self.requires("imgui/1.89.4")

    def generate(self):
        for binding in ["imgui_impl_glfw.cpp", "imgui_impl_opengl2.cpp", "imgui_impl_glfw.h", "imgui_impl_opengl2.h"]:
            conan.tools.files.copy(self, binding, self.dependencies["imgui"].cpp_info.srcdirs[0], os.path.join(self.build_folder, "bindings"))
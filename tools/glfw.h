/**
* @file glfw.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief Class for abstracting out IO via GLFW library.
*/

#pragma once

#include <GLFW/glfw3.h>

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_opengl2.h>

#include <cstdint>

#include <vector>
#include <memory>
#include <iostream>

class GLFW {
public:
  class Window {
  public:
    Window(uint32_t width, uint32_t height, const char *name = "window")
        : m_glfw(GLFW::instance())
        , m_width(width)
        , m_height(height) {

      m_window = glfwCreateWindow(width, height, name, NULL, NULL);
      if (!m_window) {
        throw std::runtime_error("Unable to create window!");
      }

      //glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

      glfwMakeContextCurrent(m_window);

      ImGui::CreateContext();

      ImGui::StyleColorsDark();

      ImGui_ImplGlfw_InitForOpenGL(m_window, true);
      ImGui_ImplOpenGL2_Init();
    }

    ~Window() {
      ImGui_ImplOpenGL2_Shutdown();
      ImGui_ImplGlfw_Shutdown();

      glfwDestroyWindow(m_window);

      ImGui::DestroyContext();
    }

    bool shouldClose() {
      return glfwWindowShouldClose(m_window);
    };

    void swapBuffers() {
      glfwSwapBuffers(m_window);
    }

    void makeContextCurrent() {
      glfwMakeContextCurrent(m_window);
    }

    int getKey(int key) const {
      return glfwGetKey(m_window, key);
    }

    int getMouseButton(int button) const {
      return glfwGetMouseButton(m_window, button);
    }

    void setCursorMode(int mode) {
      glfwSetInputMode(m_window, GLFW_CURSOR, mode);
    }

    void getCursor(float &x, float &y) {
      double xpos, ypos;
      glfwGetCursorPos(m_window, &xpos, &ypos);
      x = xpos;
      y = ypos;
    }

    void shouldClose(bool shouldClose) {
      glfwSetWindowShouldClose(m_window, shouldClose);
    }

    uint32_t width() const { return m_width; }
    uint32_t height() const { return m_height; }

  private:
    std::shared_ptr<GLFW> m_glfw;
    uint32_t m_width;
    uint32_t m_height;
    GLFWwindow *m_window = nullptr;
  };

  GLFW(const GLFW&) = delete;
  GLFW &operator=(const GLFW&) = delete;

  ~GLFW() {
    glfwTerminate();
  }

  static void pollEvents() {
    std::shared_ptr<GLFW> lock = instance();
    glfwPollEvents();
  }

private:
  static std::shared_ptr<GLFW> instance() {
    static std::weak_ptr<GLFW> instance;

    std::shared_ptr<GLFW> res = instance.lock();

    if (!res) {
        res = std::shared_ptr<GLFW>(new GLFW());
        instance = res;
    }

    return res;
  }

  GLFW() {
    if (!glfwInit()) {
      throw std::runtime_error("Unable to initialize GLFW!");
    }
  }
};

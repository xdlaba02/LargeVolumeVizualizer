#pragma once

#include <GLFW/glfw3.h>

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

      glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

      glfwMakeContextCurrent(m_window);

      m_raster.resize(width * height * 3);
    }

    ~Window() {
      glfwDestroyWindow(m_window);
    }

    bool shouldClose() {
      return glfwWindowShouldClose(m_window);
    };

    void swapBuffers() {
      glClear(GL_COLOR_BUFFER_BIT);
      glDrawPixels(m_width, m_height, GL_RGB, GL_UNSIGNED_BYTE, m_raster.data());
      glfwSwapBuffers(m_window);
    }

    void pollEvents() {
      glfwPollEvents();
    }

    int getKey(int key) {
      return glfwGetKey(m_window, key);
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

    uint8_t &raster(uint32_t x, uint32_t y, uint8_t c) {
      return m_raster[y * m_width * 3 + x * 3 + c];
    }

    uint32_t width() const { return m_width; }
    uint32_t height() const { return m_height; }

  private:
    std::shared_ptr<GLFW> m_glfw;
    uint32_t m_width;
    uint32_t m_height;
    std::vector<uint8_t> m_raster;
    GLFWwindow *m_window = nullptr;
  };

  GLFW(const GLFW&) = delete;
  GLFW &operator=(const GLFW&) = delete;

  ~GLFW() {
    glfwTerminate();
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
        throw std::runtime_error("Unable to init glfw!");
      }
    }
};

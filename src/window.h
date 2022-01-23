#pragma once

#include <GLFW/glfw3.h>

#include <cstdint>

#include <vector>

class Window {
public:
  Window(uint32_t width, uint32_t height, const char *name = "window") {
    m_width = width;
    m_height = height;

    if (!glfwInit()) {
      std::cerr << "ERROR: Unable to init glfw!\n";
      return;
    }

    m_window = glfwCreateWindow(width, height, name, NULL, NULL);
    if (!m_window) {
      std::cerr << "ERROR: Unable to create window!\n";
      return;
    }

    glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);  

    glfwMakeContextCurrent(m_window);

    m_raster.resize(width * height * 3);
  }

  ~Window() {
    glfwDestroyWindow(m_window);
    glfwTerminate();
  }

  bool shouldClose() {
    return glfwWindowShouldClose(m_window);
  };

  void swapBuffers() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
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

  uint8_t *raster() {
    return m_raster.data();
  }

  uint8_t *raster(uint32_t x, uint32_t y) {
    return &m_raster[y * m_width * 3 + x * 3];
  }

  uint32_t width() const { return m_width; }
  uint32_t height() const { return m_height; }

private:
  uint32_t m_width;
  uint32_t m_height;
  std::vector<uint8_t> m_raster;
  GLFWwindow *m_window = nullptr;
};

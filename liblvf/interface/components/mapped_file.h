#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include <fcntl.h>
#include <unistd.h>

#include <cstddef>

class MappedFile {
public:

  enum Visibility {
    SHARED,
    PRIVATE,
  };

  enum ProtectionBits {
    READ = 1 << 0,
    WRITE = 1 << 1,
  };

  MappedFile() = default;

  MappedFile(const char *filename, size_t offset, size_t size, ProtectionBits protection_bits, Visibility visibility) {
    open(filename, offset, size, protection_bits, visibility);
  }

  MappedFile(const MappedFile&) = delete;

  MappedFile &operator =(const MappedFile&) = delete;

  ~MappedFile() {
    close();
  }

  void open(const char *filename, size_t offset, size_t size, ProtectionBits protection_bits, Visibility visibility) {
    close();

    auto openmode = [](ProtectionBits protection_bits) {
      if (protection_bits == READ) { return O_RDONLY; }
      if (protection_bits & WRITE) { return O_RDWR; }
      return 0;
    };

    if (int fd = ::open(filename, openmode(protection_bits)); fd > 0) {

      auto protection = [](ProtectionBits protection_bits) {
        int prot = 0;
        if (protection_bits & READ)  { prot |= PROT_READ; }
        if (protection_bits & WRITE) { prot |= PROT_WRITE; }
        return prot;
      };

      auto flags = [](Visibility visibility) {
        switch (visibility) {
          case SHARED: return MAP_SHARED;
          case PRIVATE: return MAP_PRIVATE;
        }
        return 0;
      };

      m_data = mmap(nullptr, size, protection(protection_bits), flags(visibility), fd, offset);

      ::close(fd);

      if (m_data) {
        m_size = size;
      }
    }
  }

  void close() {
    if (m_data) {
      munmap(m_data, m_size);
      m_data = nullptr;
      m_size = 0;
    }
  }

  void *data() { return m_data; }
  const void *data() const { return m_data; }

  size_t size() const { return m_size; }

  operator bool() const { return m_data; }

private:
  void *m_data = nullptr;
  size_t m_size = 0;
};

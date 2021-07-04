//
// Created by yiak on 2021/4/30.
//

#include "file.h"

namespace svso {
namespace base {
namespace io {
namespace reader {

std::string relative_to(const std::string& path, const std::string& target )
{
    NOT_IMPLEMENTED
}

int compile_all_extensions() {
    int extension_size = sizeof(file_extensions) / sizeof(FileExtension);

    for (int i=0; i < extension_size; i++) {
        if (regcomp(&(file_extensions[i].re), file_extensions[i].def, REG_NEWLINE|REG_EXTENDED)) {
            return EXIT_FAILURE;
        }
    }
    return EXEC_SUCC;
}

FileExtension* parse_extension(const char* extension) {
    int extension_size = sizeof(file_extensions) / sizeof(FileExtension);
    regmatch_t mtched;

    for (int i=0; i < extension_size; i++) {
        if (regexec(&(file_extensions[i].re), extension, 1, &mtched, REG_NOTEOL)) {
            return file_extensions + i;
        }
    }
    return nullptr;
}

} // reader

namespace writer {

// Yaml Writer


// Protobuf Writer

} // writer

    } // base
  } // io
} // svso

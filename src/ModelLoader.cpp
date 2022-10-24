//
// Created by tztz8 on 9/26/22.
//

#include "ModelLoader.h"
#include "OpenGLHelperMethods.h"

void firstPass(SimpleModel *pModel, std::ifstream *ifstream);

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include <spdlog/spdlog.h>
#include <fstream>

SimpleModel* readOBJ(std::filesystem::path filename) {

    SPDLOG_INFO(spdlog::fmt_lib::format("readOBJ: Start load of filepath \"{}\"", filename.string()));

    SimpleModel *model;

    // Open File
    std::ifstream modelFile(filename);
    // is the file available and open
    if (!modelFile.is_open()) {
        SPDLOG_ERROR(spdlog::fmt_lib::format("readOBJ: Unable to open filepath \"{}\"", filename.string()));
        modelFile.close();
        return nullptr;
    }
    // is the file empty
    if (!modelFile.good()) {
        SPDLOG_ERROR(spdlog::fmt_lib::format("readOBJ: model file is empty: {}",
                                             filename.filename().string()));
        return nullptr;
    }

    return nullptr;
}

void firstPass(SimpleModel *pModel, std::ifstream *ifstream) {

}

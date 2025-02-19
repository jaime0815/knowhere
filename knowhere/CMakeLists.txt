#-------------------------------------------------------------------------------
# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.
#-------------------------------------------------------------------------------

include_directories(${KNOWHERE_SOURCE_DIR}/knowhere)
include_directories(${KNOWHERE_SOURCE_DIR}/thirdparty)
include_directories(${KNOWHERE_SOURCE_DIR}/thirdparty/faiss)
include_directories(${KNOWHERE_SOURCE_DIR}/thirdparty/DiskANN)
include_directories(${KNOWHERE_SOURCE_DIR}/knowhere/utils)

if ( KNOWHERE_WITH_DISKANN )
        include_directories(${KNOWHERE_SOURCE_DIR}/thirdparty/DiskANN)  
endif()

set(KNOWHERE_THIRDPARTY_SRC ${KNOWHERE_SOURCE_DIR}/thirdparty)
set(external_srcs
        common/Exception.cpp
        common/Timer.cpp
        common/Log.cpp
        common/Utils.cpp
        ${KNOWHERE_THIRDPARTY_SRC}/easyloggingpp/easylogging++.cc
        )

set(config_srcs
        archive/KnowhereConfig.cpp
        archive/BruteForce.cpp
        )

set(vector_index_srcs
        index/VecIndexFactory.cpp
        index/vector_index/adapter/VectorAdapter.cpp
        index/vector_index/helpers/FaissIO.cpp
        index/vector_index/helpers/Slice.cpp
        index/vector_index/ConfAdapter.cpp
        index/vector_index/ConfAdapterMgr.cpp
        index/vector_index/FaissBaseBinaryIndex.cpp
        index/vector_index/FaissBaseIndex.cpp
        index/vector_index/IndexBinaryIDMAP.cpp
        index/vector_index/IndexBinaryIVF.cpp
        index/vector_index/IndexIDMAP.cpp
        index/vector_index/IndexIVF.cpp
        index/vector_index/IndexIVFPQ.cpp
        index/vector_index/IndexIVFSQ.cpp
        index/vector_index/IndexAnnoy.cpp
        index/vector_index/IndexHNSW.cpp
        index/vector_index/Statistics.cpp
        )

if ( KNOWHERE_WITH_DISKANN )
    set(vector_index_srcs
            index/vector_index/IndexDiskANN.cpp
            index/vector_index/IndexDiskANNConfig.cpp
            ${vector_index_srcs}
            )
    set(KNOWHERE_INCLUDE_DIRS
            ${KNOWHERE_INCLUDE_DIRS}
            ${DISKANN_INCLUDE_DIR}
            )
endif()

set(vector_offset_index_srcs
        index/vector_offset_index/OffsetBaseIndex.cpp
        index/vector_offset_index/IndexIVF_NM.cpp
        )

set(depend_libs
        faiss
        pthread
        knowhere_utils
        )

if (KNOWHERE_WITH_DISKANN)
      set(depend_libs
             ${depend_libs}
             diskann
             aio
             )
endif()

if (KNOWHERE_GPU_VERSION)
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
    set(cuda_lib
            cudart
            cublas
            )
    set(depend_libs ${depend_libs}
            ${cuda_lib}
            )

    set(vector_index_srcs ${vector_index_srcs}
            index/vector_index/gpu/IndexGPUIDMAP.cpp
            index/vector_index/gpu/IndexGPUIVF.cpp
            index/vector_index/gpu/IndexGPUIVFPQ.cpp
            index/vector_index/gpu/IndexGPUIVFSQ.cpp
            index/vector_index/helpers/Cloner.cpp
            index/vector_index/helpers/FaissGpuResourceMgr.cpp
            )

    set(vector_offset_index_srcs ${vector_offset_index_srcs}
            index/vector_offset_index/gpu/IndexGPUIVF_NM.cpp
            )
endif ()

if (NOT TARGET knowhere)
    add_library(
            knowhere SHARED
            ${external_srcs}
            ${config_srcs}
            ${vector_index_srcs}
            ${vector_offset_index_srcs}
    )
    target_include_directories(knowhere PUBLIC ${KNOWHERE_SOURCE_DIR}/knowere)
endif ()

target_link_libraries(knowhere ${depend_libs})

set (KNOWHERE_INCLUDE_DIRS
        ${KNOWHERE_SOURCE_DIR}
        ${KNOWHERE_SOURCE_DIR}/thirdparty
        ${FAISS_INCLUDE_DIR}
        ${OpenBLAS_INCLUDE_DIR}
        ${LAPACK_INCLUDE_DIR}
        )

target_include_directories(knowhere PUBLIC ${KNOWHERE_INCLUDE_DIRS})

install(TARGETS knowhere DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY "${KNOWHERE_SOURCE_DIR}/knowhere/"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include/knowhere"
        FILES_MATCHING
        PATTERN "*.h"
)

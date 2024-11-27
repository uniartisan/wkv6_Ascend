#!/bin/bash
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
export ASCEND_HOME_DIR=/usr/local/Ascend/ascend-toolkit/latest #替换为当前CANN包路径
source $ASCEND_HOME_DIR/../set_env.sh

SHORT=r:,v:,
LONG=run-mode:,soc-version:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
while :
do
    case "$1" in
        (-r | --run-mode )
            RUN_MODE="$2"
            shift 2;;
        (-v | --soc-version )
            SOC_VERSION="$2"
            shift 2;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done

rm -rf build
mkdir build
cd build

# in case of running op in simulator, use stub so instead
if [ "${RUN_MODE}" = "sim" ]; then
    export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed 's/\/.*\/runtime\/lib64://g')
    export LD_LIBRARY_PATH=$ASCEND_HOME_DIR/runtime/lib64/stub:$LD_LIBRARY_PATH

    if [ ! $CAMODEL_LOG_PATH ]; then
        export CAMODEL_LOG_PATH=./ # default log save in build dir
    else
        export CAMODEL_LOG_PATH=../$CAMODEL_LOG_PATH
        rm -rf $CAMODEL_LOG_PATH
        mkdir -p $CAMODEL_LOG_PATH
    fi
fi

if [ "${RUN_MODE}" = "cpu" ]; then
    export CAMODEL_LOG_PATH=./ # cpu run mode set fixed log path
fi

source $ASCEND_HOME_DIR/bin/setenv.bash
export LD_LIBRARY_PATH=${ASCEND_HOME_DIR}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH

cmake  -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION}  -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_DIR} ..
make -j16
source $ASCEND_HOME_DIR/../set_env.sh

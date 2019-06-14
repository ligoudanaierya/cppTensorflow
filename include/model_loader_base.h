#ifndef CPPTENSORFLOW_MODEL_LOADER_BASH_H
#define CPPTENSORFLOW_MODEL_LOADER_BASE_H

#include <iostream>
#include <vector>
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/cc/ops/array_ops.h"  
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
namespace tf_model{
    class DataInputBase{
    public:
        DataInputBase(){};
        virtual ~DataInputBase(){};
        virtual std::vector<cv::Mat> get_clip(const int)=0;
        virtual void data_process(std::vector<cv::Mat>&)=0;
        int s_index;
        bool is_end;
        std::vector<cv::Mat> video;

    };
    class FeatureAdapterBase{
    public:
        FeatureAdapterBase(){};
        virtual ~FeatureAdapterBase(){};
        virtual void assign(std::string,std::vector<cv::Mat>&)=0;
        std::vector<std::pair<std::string,tensorflow::Tensor> > input;
    };
    class ModelLoaderBase{
        public:
            ModelLoaderBase(){};
            virtual ~ModelLoaderBase(){};
            virtual int load(tensorflow::Session*,const std::string)=0;
            virtual int predict(tensorflow::Session*,const FeatureAdapterBase&, const std::string,std::vector<Eigen::Tensor<float,1>>&)=0;
            tensorflow::GraphDef graphdef;
    };
}

#endif 

#ifndef CPPTENSORFLOW_MODEL_LOADER_H
#define CPPTENSORFLOW_MODEL_LOADER_H

#include "model_loader_base.h"

namespace tf_model{
    class DataInput:public DataInputBase{
        public:
            DataInput(std::string video_path);
            DataInput(std::vector<cv::Mat> &images);
            ~DataInput();
            std::vector<cv::Mat>get_clip(const int) override;
            void data_process(std::vector<cv::Mat>&) override;
            const int crop_size = 224;
            const int nums_per_clip = 64;
    };
    class FeatureAdapter:public FeatureAdapterBase{
        public:
            FeatureAdapter();
            ~FeatureAdapter();
            void assign(std::string tname,std::vector<cv::Mat>&) override;
    };
    class ModelLoader:public ModelLoaderBase {
        public:
            ModelLoader();
            ~ModelLoader();
            int load(tensorflow::Session*,const std::string) override;
            int predict(tensorflow::Session*,const FeatureAdapterBase&,const std::string,std::vector<Eigen::Tensor<float,1>>&) override;
    };
}
int i3dpredict(std::string,std::string model_path="../pbmodel/model.pb");
int i3dpredict(std::vector<cv::Mat>&,std::string model_path="../pbmodel/model.pb");
#endif

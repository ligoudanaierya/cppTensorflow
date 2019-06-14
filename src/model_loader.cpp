//#include<iostream>
//#include<vector>
//#include<string>
#include"model_loader.h"


// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
#define crop_size 224
#define nums_clip 64
#define _cha 3
//#undef CV_MAKETYPE(CV_32F,3)
namespace tf_model{


    DataInput::DataInput(std::string video_path)
    {
        cv::VideoCapture capture(video_path);
        if(!capture.isOpened())
        {
            std::cout<<"failed open"<<std::endl;
        }
        long totalFrameNum = static_cast<long>(capture.get(cv::CAP_PROP_FRAME_COUNT));
        if(totalFrameNum==0)
        {
            std::cout<<"Could not initialize capturing"<<video_path<<std::endl;
        }

        cv::Mat frame;
        while(1)
        {
            capture>>frame;
            if(frame.empty())
            {
                break;
            }
            video.push_back(frame.clone());
        }
        s_index = 0;
        is_end = false;
    }
    DataInput::DataInput(std::vector<cv::Mat>&images)
    {
        for(int i=0;i<images.size();i++)
        {
            video.push_back(images[i].clone());
        }
        s_index = 0;
        is_end = false;
    }
    DataInput::~DataInput(){}
    std::vector<cv::Mat> DataInput::get_clip(const int nums_per_clip)
    {
        std::vector<cv::Mat> res;
        if(video.size()-s_index<= nums_per_clip)
        {
           for (int i=0;i<nums_per_clip;i++)
           {
                int j= i; 
                if(i>=video.size())
                    j = video.size()-s_index-1;
                           //Mat img = images[j];
                           //imwrite(to_string(i)+".jpg",images[j]);
                           //cout<<"j: "<<j<<endl;
                res.push_back(video[j].clone());
            }
            is_end = true ;
            s_index+=nums_per_clip;
            data_process(res);
            return res;
          }
         for(int i =s_index;i<s_index+nums_per_clip;i++)
         {
                       //cout<<i<<endl;
            cv::Mat img = video[i];
            res.push_back(img.clone());
         }
         is_end = false;
         s_index +=nums_per_clip;
         data_process(res);
         return res;

    }
    void DataInput::data_process(std::vector<cv::Mat>& res)
    {
        cv::Mat tmp;
        for(int i =0;i<res.size();i++)
        {
            tmp = res[i];
            if(tmp.cols>tmp.rows)
            {
                float scale = static_cast<float>(256)/static_cast<float>(tmp.rows);
                cv::resize(tmp,res[i],cv::Size(static_cast<int>(tmp.cols*scale+1),crop_size));
            }
            else
            {
                float scale = static_cast<float>(256)/static_cast<float>(tmp.cols);
                cv::resize(tmp,res[i],cv::Size(crop_size,static_cast<int>(tmp.rows*scale+1)));
            }
            int crop_c = static_cast<int>((res[i].cols-crop_size)/2);
            int crop_r = static_cast<int>((res[i].rows-crop_size)/2);
            res[i] = res[i](cv::Rect(crop_c,crop_r,crop_size,crop_size));
        }
    }

    FeatureAdapter::FeatureAdapter(){
    }
    FeatureAdapter::~FeatureAdapter(){
    }

void FeatureAdapter::assign(std::string tname,std::vector<cv::Mat>& images)

{
   int nums = images.size();
   if(nums!=nums_clip)
   {
       std::cout<<"WARING : input clip size is wrong"<<nums<<std::endl;
   }
   tensorflow::Tensor phase_train(tensorflow::DT_BOOL, tensorflow::TensorShape());
   phase_train.scalar<bool>()() = false;
   tensorflow:: Tensor intensor(tensorflow::DT_FLOAT,tensorflow::TensorShape({1,static_cast<long long int>(images.size()),crop_size,crop_size,_cha}));
   auto tensormap = intensor.tensor<float,5>();
   for(int i=0;i<images.size();i++)
   {
       images[i].convertTo(images[i],CV_32FC3);
       int l_height = images[i].size().height;
       int l_width = images[i].size().width;
       int l_depth = images[i].channels();
       const float* data = (float*)images[i].data;
       for(int y=0;y<l_height;y++)
       {
           const float* dataRow = data+(y*l_width*l_depth);
           for (int x=0;x<l_width;x++)
           {
              const float* datapixal = dataRow+(x*l_depth);
              for(int c=0;c<l_depth;c++)
              {
                  const float* datavalue = datapixal+c;
                  tensormap(0,i,y,x,c)=*datavalue;
              }
           }
       }
   }
   input.push_back(std::pair<std::string,tensorflow::Tensor>(tname,intensor));
   input.push_back(std::pair<std::string,tensorflow::Tensor>("Placeholder_2",phase_train));
}

ModelLoader::ModelLoader(){}
ModelLoader::~ModelLoader(){}

int ModelLoader::load(tensorflow::Session* session,const std::string model_path){
    tensorflow::Status status_load = tensorflow::ReadBinaryProto(tensorflow::Env::Default(),model_path,&graphdef);
    if(!status_load.ok())
    {
        std::cout<<"ERROR:Loading model failed.."<<model_path<<std::endl;
        std::cout<<status_load.ToString()<<std::endl;
        return -1;
    }
    tensorflow::Status status_create = session->Create(graphdef);
    if(!status_create.ok()){
        std::cout<<"ERROR: Creating graph in session failed..."<<status_create.ToString() << std::endl;
        return -1;
    }
    return 0;
}
int ModelLoader::predict(tensorflow::Session* session,const FeatureAdapterBase& input_feature,const std::string output_node,std::vector<Eigen::Tensor<float,1>>&tensors)
{
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::Status status = session->Run(input_feature.input,{output_node},{},&outputs);
    if(!status.ok())
    {
        std::cout<<"ERROR: predicetion failed.."<<status.ToString()<<std::endl;
        return -1;
    }
    std::cout<<"Output tensor size: "<<outputs.size()<<std::endl;
    for(std::size_t i = 0; i<outputs.size();i++)
    {
        std::cout<<outputs[i].DebugString();
    }
    std::cout<<std::endl;
    tensorflow::Tensor t = outputs[0];
    int ndim = t.shape().dims();
    auto tmap = t.tensor<float,2>();
    int output_dim = t.shape().dim_size(1);
    std::vector<float> tout;
    int output_class_id = -1;
    Eigen::Tensor<float,1> emap(output_dim);
    for(int j =0;j<output_dim;j++)
    {
        emap(j)=tmap(0,j);

    }
    emap = emap.exp(); 
    Eigen::Tensor<float, 0> e_x_sum = emap.sum();
    emap = emap/e_x_sum();
    tensors.push_back(emap);
    double output_prob = 0.0;
    for(int j = 0;j<output_dim;j++)
    {
        if(emap(j)>=output_prob)
        {
            output_class_id = j;
            output_prob = emap(j);
        }
    }
    std::cout << "Clip Class id: " << output_class_id <<" Clip Prob is: "<<output_prob<< std::endl;
    //std::cout << "Clip value is: " << output_prob << std::endl;
    return 0;
}
}
int i3dpredict(std::string video_path,std::string model_path)
{
    tf_model::DataInput data(video_path);
    std::string input_tensor_name = "rgbinput";
    std::string output_tensor_name = "RGB/inception_i3d/output";
    tensorflow::Session* session;
    tensorflow::Status status = tensorflow::NewSession(tensorflow::SessionOptions(),&session);
    if(!status.ok())
    {
        std::cout<<status.ToString()<<"\n";
        return -1;
    }
    tf_model::FeatureAdapter input_feat;
    tf_model::ModelLoader model;
    if(0!=model.load(session,model_path))
    {
        std::cout<<"ERROE: Model Loading failed"<<std::endl;
        return -1;
    }
    std::vector<Eigen::Tensor<float,1>> tensors;
    std::vector<cv::Mat> res;
    while(!data.is_end)
    {
        input_feat.input.clear();
        res = data.get_clip(nums_clip);
        input_feat.assign(input_tensor_name,res);
        if(0!=model.predict(session,input_feat,output_tensor_name,tensors))
        {
            std::cout<<"ERROR: Prediction failed..."<<std::endl;
            return -1;
        }
    }
    if(0==tensors.size())
    {
        std::cout<<"ERROR: No Result"<<std::endl;
        return -1;
    }
    std::vector<float> prob(tensors[0].dimension(0),0);
    int _class;
    
    for(size_t i =0;i<tensors.size();i++)
    {
        for(size_t j=0;j<tensors[i].dimension(0);j++)
        {
            prob[j]+=tensors[i](j);
        }
    }
    std::vector<float>::iterator biggest = std::max_element(std::begin(prob), std::end(prob));
    _class = std::distance(std::begin(prob),biggest);
    std::cout<<"The Class is "<<_class<<" The AVEProb is "<<*biggest/tensors.size()<<std::endl;
    return _class;

}

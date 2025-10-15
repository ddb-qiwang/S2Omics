import argparse
from s2omics.p1_histology_preprocess import histology_preprocess
from s2omics.p2_superpixel_quality_control import superpixel_quality_control
from s2omics.p3_feature_extraction import histology_feature_extraction
from s2omics.single_section.p7_cell_label_broadcasting import label_broadcasting

def get_args():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--WSI_datapath', type=str)
    parser.add_argument('--SO_datapath', type=str)
    parser.add_argument('--WSI_save_folder', type=str)
    parser.add_argument('--SO_save_folder', type=str)
    parser.add_argument('--WSI_cache_path', type=str, default='')
    parser.add_argument('--SO_cache_path', type=str, default='')
    parser.add_argument('--need_preprocess', type=bool, default=False)
    parser.add_argument('--need_feature_extraction', type=bool, default=False)
    parser.add_argument('--foundation_model', type=str, default='uni')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/uni/')
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()

def main():
    args = get_args()
    
    if args.need_preprocess:
        histology_preprocess(args.WSI_datapath, show_image=False)
        histology_preprocess(args.SO_datapath, show_image=False)
        superpixel_quality_control(args.WSI_datapath, args.WSI_save_folder, show_image=False)
        superpixel_quality_control(args.SO_datapath, args.SO_save_folder, show_image=False)
    
    if args.need_feature_extraction:
        histology_feature_extraction(args.WSI_datapath, args.WSI_save_folder,
                                     foundation_model=args.foundation_model,
                                     ckpt_path=args.ckpt_path,
                                     device=args.device,
                                     down_samp_step=1)   
        if not args.WSI_datapath == args.SO_datapath:
            histology_feature_extraction(args.SO_datapath, args.SO_save_folder,
                                         foundation_model=args.foundation_model,
                                         ckpt_path=args.ckpt_path,
                                         device=args.device,
                                         down_samp_step=1)   

    label_broadcasting(WSI_datapath=args.WSI_datapath,
                       SO_datapath=args.SO_datapath,
                       WSI_save_folder=args.WSI_save_folder,
                       SO_save_folder=args.SO_save_folder,
                       WSI_cache_path=args.WSI_cache_path,
                       SO_cache_path=args.SO_cache_path,
                       foundation_model=args.foundation_model,
                       device=args.device)
                       
if __name__ == '__main__':
    main()
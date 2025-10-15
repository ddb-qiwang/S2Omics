import argparse
from s2omics.p1_histology_preprocess import histology_preprocess
from s2omics.p2_superpixel_quality_control import superpixel_quality_control
from s2omics.p3_feature_extraction import histology_feature_extraction
from s2omics.single_section.p4_get_histology_segmentation import get_histology_segmentation
from s2omics.single_section.p5_merge_over_clusters import merge_over_clusters
from s2omics.single_section.p6_roi_selection_rectangle import roi_selection_for_single_section
## if need to select circle-shaped ROI, please
# from s2omics.single_section.p6_roi_selection_circle import roi_selection_for_single_section

def get_args():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--save_folder', type=str)
    parser.add_argument('--foundation_model', type=str, default='uni')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/uni/')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--down_samp_step', type=int, default=10)
    parser.add_argument('--clustering_method', type=str, default='kmeans')
    parser.add_argument('--n_clusters', type=int, default=20)
    parser.add_argument('--target_n_clusters', type=int, default=15)
    parser.add_argument('--roi_size', type=float, nargs='+', default=[6.5,6.5])
    parser.add_argument('--num_roi', type=int, default=0) # 0 means automatically determine the optimal number of ROIs
    parser.add_argument('--fusion_weights', type=float, nargs='+', default=[0.33,0.33,0.33]) # [size_score_wight, coverage_score_weight, balance_score_weight], sum of them should be equal to 1
    parser.add_argument('--emphasize_clusters', type=int, nargs='+', default=[]) # [emphasize_clusters]
    parser.add_argument('--discard_clusters', type=int, nargs='+', default=[]) # [discard_clusters]
    parser.add_argument('--prior_preference', type=int, default=2) 

    return parser.parse_args()

def main():
    args = get_args()

    histology_preprocess(args.prefix, show_image=False)
    
    superpixel_quality_control(args.prefix, args.save_folder, show_image=False)
    
    histology_feature_extraction(args.prefix, args.save_folder,
                                 foundation_model=args.foundation_model,
                                 ckpt_path=args.ckpt_path,
                                 device=args.device,
                                 down_samp_step=args.down_samp_step) 

    get_histology_segmentation(args.prefix, args.save_folder,
                               foundation_model=args.foundation_model, 
                               down_samp_step=args.down_samp_step, 
                               clustering_method=args.clustering_method,
                               n_clusters=args.n_clusters)

    merge_over_clusters(args.prefix, args.save_folder,
                        target_n_clusters=args.target_n_clusters)

    roi_selection_for_single_section(args.prefix, args.save_folder,
                                     down_samp_step=args.down_samp_step,
                                     roi_size=args.roi_size,
                                     num_roi=args.num_roi, #0 refers to automatiacally determine the number of ROI
                                     fusion_weights=args.fusion_weights,
                                     emphasize_clusters=args.emphasize_clusters, 
                                     discard_clusters=args.discard_clusters,
                                     prior_preference=args.prior_preference)

                       
if __name__ == '__main__':
    main()
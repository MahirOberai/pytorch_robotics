import image
import segmentation_helper as sh

#rgb = image.read_rgb('./dataset/train/286_rgb.png')
pred = image.read_mask('./dataset/val/pred/1_pred.png')
gt = image.read_mask('./dataset/val/gt/1_gt.png')
#sh.show_mask(pred)
sh.show_mask(gt)



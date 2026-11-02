图片可直接存放于此，也可通过 --input 指向其他仓库的 JPEGImages，无需重复复制。
VOC 划分通过 --split /path/to/ImageSets/test.txt 指定；每行保留完整文件名主干（包含空格）。
GT 使用 --gt /path/to/SegmentationClass，必须是原图同尺寸的单通道/P模式类别索引 PNG。

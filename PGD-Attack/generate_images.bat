for %%x in (.005 .02) do (
    for %%y in (5 10 15) do (
        python3.11.exe .\PGD-Attack\create_adversarial_images.py --PGD_image_count 100 --PGD_niter %%y --PGD_epsilon %%x --PGD_stepsize .01 --batch_size 100 --pretrained_path .\ResNet-50-CBAM-PyTorch\pretrained_weights\cifar_10_dataset_clean_models\resnet_cbam\resnet_cbam\20_epoch_model.pt --data_folder .\datasets\cifar-10\ --PGD_save_path "./test_images"
    )
)

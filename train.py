from original import *
import shutil, glob
from easyfuncs import download_from_url, CachedModels
os.makedirs("dataset",exist_ok=True)
model_library = CachedModels()

with gr.Blocks(title="TRAIN",theme=gr.themes.Base()) as app:
    with gr.Row():
        gr.Markdown("RVC TRAINING MODE")
    with gr.Tabs():
        with gr.TabItem(i18n("Train")):
            gr.Markdown(value=i18n(""))
            with gr.Row():
                exp_dir1 = gr.Textbox(label=i18n("Model Name"), value="test-model")
                sr2 = gr.Dropdown(
                    label=i18n("Sample Rate & Pretrain"),
                    choices=["32k", "40k", "48k"],
                    value="32k",
                    interactive=True,
                )
                version19 = gr.Radio(
                    label=i18n("Version 2 only here"),
                    choices=["v2"],
                    value="v2",
                    interactive=False,
                    visible=False,
                )
                np7 = gr.Slider(
                    minimum=0,
                    maximum=config.n_cpu,
                    step=1,
                    label=i18n("CPU Threads"),
                    value=int(np.ceil(config.n_cpu / 2.5)),
                    interactive=True,
                )
            with gr.Group():
                gr.Markdown(value=i18n(""))
                with gr.Row():
                    trainset_dir4 = gr.Textbox(
                        label=i18n("Path to Dataset"), value="dataset"
                    )
                    with gr.Accordion('Upload Dataset (alternative)', open=False, visible=True):
                        file_thin = gr.Files(label='Dataset') # transfers files to the dataset dir, lol # much coding -ila
                        show = gr.Textbox(label='Status')
                        transfer_button = gr.Button('Upload Dataset to the folder', variant="primary")
                        transfer_button.click(
                            fn=transfer_files,
                            inputs=[file_thin],
                            outputs=show,
                        )

            with gr.Group():
                gr.Markdown(value=i18n(""))
                with gr.Row():
                    save_epoch10 = gr.Slider(
                        minimum=1,
                        maximum=250,
                        step=1,
                        label=i18n("Save frequency"),
                        value=50,
                        interactive=True,
                    )
                    total_epoch11 = gr.Slider(
                        minimum=2,
                        maximum=10000,
                        step=1,
                        label=i18n("Total Epochs"),
                        value=300,
                        interactive=True,
                    )
                    batch_size12 = gr.Slider(
                        minimum=1,
                        maximum=16,
                        step=1,
                        label=i18n("Batch Size"),
                        value=default_batch_size,
                        interactive=True,
                    )
                    if_save_every_weights18 = gr.Radio(
                        label=i18n("Create model with save frequency"),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("是"),
                        interactive=True,
                    )

            with gr.Accordion('Advanced Settings', open=False, visible=True):
                with gr.Row(): 
                    with gr.Group():
                        spk_id5 = gr.Slider(
                                minimum=0,
                                maximum=4,
                                step=1,
                                label=i18n("Speaker ID"),
                                value=0,
                                interactive=True,
                            )
                        if_f0_3 = gr.Radio(
                        label=i18n("Pitch Guidance"),
                        choices=[True, False],
                        value=True,
                        interactive=True,
                    )
                        gpus6 = gr.Textbox(
                                label=i18n("GPU ID (Leave 0 if you have only one GPU, use 0-1 for multiple GPus)"),
                                value=gpus,
                                interactive=True,
                                visible=F0GPUVisible,
                            )
                        gpu_info9 = gr.Textbox(
                                label=i18n("GPU Model"),
                                value=gpu_info,
                                visible=F0GPUVisible,
                            )
                        gpus16 = gr.Textbox(
                        label=i18n("Enter cards to be used (Leave 0 if you have only one GPU, use 0-1 for multiple GPus)"),
                        value=gpus if gpus != "" else "0",
                        interactive=True,
                        )
                        with gr.Group():
                            if_save_latest13 = gr.Radio(
                                label=i18n("Save last ckpt as final Model"),
                                choices=[i18n("是"), i18n("否")],
                                value=i18n("是"),
                                interactive=True,
                            )
                            if_cache_gpu17 = gr.Radio(
                                label=i18n("Cache data to GPU (Only for datasets under 8 minutes)"),
                                choices=[i18n("是"), i18n("否")],
                                value=i18n("否"),
                                interactive=True,
                            )
                            f0method8 = gr.Radio(
                                    label=i18n("Feature Extraction Method"),
                                    choices=["rmvpe", "rmvpe_gpu"],
                                    value="rmvpe_gpu",
                                    interactive=True,
                                )
                            gpus_rmvpe = gr.Textbox(
                                    label=i18n(
                                        "rmvpe_gpu will use your GPU instead of the CPU for the feature extraction"
                                    ),
                                    value="%s-%s" % (gpus, gpus),
                                    interactive=True,
                                    visible=F0GPUVisible,
                                )
                            f0method8.change(
                                fn=change_f0_method,
                                inputs=[f0method8],
                                outputs=[gpus_rmvpe],
                            )        

            with gr.Row():
                pretrained_G14 = gr.Textbox(
                    label="Pretrained G",
                    choices=list(pretrained_G_files.values()),
                    value=pretrained_G_files.get('f0G32.pth', ''),
                    visible=False,
                    interactive=True,
                )
                pretrained_D15 = gr.Textbox(
                    label="Pretrained D",
                    choices=list(pretrained_D_files.values()),
                    value=pretrained_D_files.get('f0D32.pth', ''),
                    visible=False,
                    interactive=True,
                )
                sr2.change(
                    change_sr2,
                    [sr2, if_f0_3, version19],
                    [pretrained_G14, pretrained_D15],
                )
                version19.change(
                    change_version19,
                    [sr2, if_f0_3, version19],
                    [pretrained_G14, pretrained_D15, sr2],
                )
                if_f0_3.change(
                    change_f0,
                    [if_f0_3, sr2, version19],
                    [f0method8, gpus_rmvpe, pretrained_G14, pretrained_D15],
                )
            
            with gr.Group():
                def one_click_train(trainset_dir4, exp_dir1, sr2, gpus6, np7, f0method8, if_f0_3, version19, gpus_rmvpe):
                    preprocess_dataset(trainset_dir4, exp_dir1, sr2, np7)
                    extract_f0_feature(gpus6, np7, f0method8, if_f0_3, exp_dir1, version19, gpus_rmvpe)
                    train_index(exp_dir1, version19)
                    click_train(exp_dir1, sr2, if_f0_3, spk_id5, save_epoch10, total_epoch11, batch_size12, if_save_latest13, 
                                pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18, version19)
                with gr.Row():
                    with gr.TabItem("One-Click Training"):
                        but5 = gr.Button('Train', variant="primary")
                        info = gr.Textbox(label=i18n("Output"), value="", max_lines=5, lines=5)
                        but5.click(
                            one_click_train,
                            [trainset_dir4, exp_dir1, sr2, gpus6, np7, f0method8, if_f0_3, version19, gpus_rmvpe]
                        )

                    with gr.TabItem("Manual Training"):
                        but1 = gr.Button(i18n("1. Process Data"), variant="primary")
                        but2 = gr.Button(i18n("2. Feature Extraction"), variant="primary")
                        but4 = gr.Button(i18n("3. Train Index"), variant="primary")
                        but3 = gr.Button(i18n("4. Train Model"), variant="primary")
                        info = gr.Textbox(label=i18n("Output"), value="", max_lines=5, lines=5)
                        but1.click(
                            preprocess_dataset,
                                [trainset_dir4, exp_dir1, sr2, np7],
                                [info],
                                api_name="train_preprocess",
                            )
                        but2.click(
                            extract_f0_feature,
                                [
                                    gpus6,
                                    np7,
                                    f0method8,
                                    if_f0_3,
                                    exp_dir1,
                                    version19,
                                    gpus_rmvpe,
                                ],
                                [info],
                                api_name="train_extract_f0_feature",
                        )
                        but4.click(train_index, [exp_dir1, version19], info)
                        but3.click(
                            click_train,
                            [
                                exp_dir1,
                                sr2,
                                if_f0_3,
                                spk_id5,
                                save_epoch10,
                                total_epoch11,
                                batch_size12,
                                if_save_latest13,
                                pretrained_G14,
                                pretrained_D15,
                                gpus16,
                                if_cache_gpu17,
                                if_save_every_weights18,
                                version19,
                            ],
                            info,
                            api_name="train_start",
                        )
                        but4.click(train_index, [exp_dir1, version19], info)
    if config.iscolab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )

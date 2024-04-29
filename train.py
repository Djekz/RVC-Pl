from original import *
import shutil, glob
from easyfuncs import download_from_url, CachedModels
os.makedirs("dataset",exist_ok=True)
model_library = CachedModels()

with gr.Blocks(title="TRAIN",theme=gr.themes.Base()) as app:
    with gr.Row():
        gr.Markdown("RVC TRAINING MODE")
    with gr.Tabs():
        with gr.TabItem("Train"):
            with gr.Row():
                with gr.Column():
                    training_name = gr.Textbox(label="Name your model", value="My-Voice",placeholder="My-Voice")
                    np7 = gr.Slider(
                        minimum=0,
                        maximum=config.n_cpu,
                        step=1,
                        label="Number of CPU processes used to extract pitch features",
                        value=int(np.ceil(config.n_cpu / 1.5)),
                        interactive=True,
                    )
                    sr2 = gr.Radio(
                        label="Sampling Rate",
                        choices=["40k", "32k"],
                        value="32k",
                        interactive=True,
                        visible=False
                    )
                    if_f0_3 = gr.Radio(
                        label="Will your model be used for singing? If not, you can ignore this.",
                        choices=[True, False],
                        value=True,
                        interactive=True,
                        visible=False
                    )
                    version19 = gr.Radio(
                        label="Version",
                        choices=["v1", "v2"],
                        value="v2",
                        interactive=True,
                        visible=False,
                    )
                    dataset_folder = gr.Textbox(
                        label="dataset folder", value='dataset'
                    )
                    easy_uploader = gr.Files(label="Drop your audio files here",file_types=['audio'])
                    easy_uploader.upload(inputs=[dataset_folder],outputs=[],fn=lambda folder:os.makedirs(folder,exist_ok=True))
                    easy_uploader.upload(
                        fn=lambda files,folder: [shutil.copy2(f.name,os.path.join(folder,os.path.split(f.name)[1])) for f in files] if folder != "" else gr.Warning('Please enter a folder name for your dataset'),
                        inputs=[easy_uploader, dataset_folder], 
                        outputs=[])
                    gpus6 = gr.Textbox(
                        label="Enter the GPU numbers to use separated by -, (e.g. 0-1-2)",
                        value=gpus,
                        interactive=True,
                        visible=F0GPUVisible,
                    )
                    gpu_info9 = gr.Textbox(
                        label="GPU Info", value=gpu_info, visible=F0GPUVisible
                    )
                    spk_id5 = gr.Slider(
                        minimum=0,
                        maximum=4,
                        step=1,
                        label="Speaker ID",
                        value=0,
                        interactive=True,
                        visible=False
                    )
                    but1.click(
                        preprocess_dataset,
                        [dataset_folder, training_name, sr2, np7],
                        [info1],
                        api_name="train_preprocess",
                    ) 
                with gr.Column():
                    f0method8 = gr.Radio(
                        label="F0 extraction method",
                        choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
                        value="rmvpe_gpu",
                        interactive=True,
                    )
                    gpus_rmvpe = gr.Textbox(
                        label="GPU numbers to use separated by -, (e.g. 0-1-2)",
                        value="%s-%s" % (gpus, gpus),
                        interactive=True,
                        visible=F0GPUVisible,
                    )
                    f0method8.change(
                        fn=change_f0_method,
                        inputs=[f0method8],
                        outputs=[gpus_rmvpe],
                    )
                    but2.click(
                        extract_f0_feature,
                        [
                            gpus6,
                            np7,
                            f0method8,
                            if_f0_3,
                            training_name,
                            version19,
                            gpus_rmvpe,
                        ],
                        [info2],
                        api_name="train_extract_f0_feature",
                    )
                with gr.Column():
                    total_epoch11 = gr.Slider(
                        minimum=2,
                        maximum=1000,
                        step=1,
                        label="Epochs (more epochs may improve quality but takes longer)",
                        value=150,
                        interactive=True,
                    )
                    with gr.Accordion(label="Manual Train", open=False):
                        with gr.Row():
                            but1 = gr.Button("1. Process", variant="primary")
                            info1 = gr.Textbox(label="Information", value="",visible=True)
                        with gr.Row():
                            but2 = gr.Button("2. Extract Features", variant="primary")
                            info2 = gr.Textbox(label="Information", value="", max_lines=8)
                         with gr.Row():
                            but4 = gr.Button("3. Train Index", variant="primary")
                            but4.click(train_index, [training_name, version19], info3)
                        with gr.Row():
                            but3 = gr.Button("4. Train Model", variant="primary")
                            but3.click(click_train, [training_name,sr2,if_f0_3,spk_id5,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,if_cache_gpu17,gpus16,if_save_every_weights18,version19], info3,api_name="train_start")
                        with gr.Row():
                            info3 = gr.Textbox(label="Information", value="", max_lines=10)
                    with gr.Accordion(label="one click Train", open=False):
                        with gr.Row():
                            but5 = gr.Button("1 Click Training", variant="primary", visible=False)           
                            but5.click(train1key, [training_name,sr2,if_f0_3,dataset_folder,spk_id5,np7,f0method8,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16,if_cache_gpu17,if_save_every_weights18,version19,gpus_rmvpe], info3,api_name="train_start_all")

                    with gr.Accordion(label="General Settings", open=False):
                        gpus16 = gr.Textbox(
                            label="GPUs separated by -, (e.g. 0-1-2)",
                            value="0",
                            interactive=True,
                            visible=True
                        )
                        save_epoch10 = gr.Slider(
                            minimum=1,
                            maximum=50,
                            step=1,
                            label="Weight Saving Frequency",
                            value=25,
                            interactive=True,
                        )
                        batch_size12 = gr.Slider(
                            minimum=1,
                            maximum=40,
                            step=1,
                            label="Batch Size",
                            value=default_batch_size,
                            interactive=True,
                        )
                        if_save_latest13 = gr.Radio(
                            label="Only save the latest model",
                            choices=["yes", "no"],
                            value="yes",
                            interactive=True,
                            visible=False
                        )
                        if_cache_gpu17 = gr.Radio(
                            label="If your dataset is UNDER 10 minutes, cache it to train faster",
                            choices=["yes", "no"],
                            value="no",
                            interactive=True,
                        )
                        if_save_every_weights18 = gr.Radio(
                            label="Save small model at every save point",
                            choices=["yes", "no"],
                            value="yes",
                            interactive=True,
                        )
                    with gr.Row():
                        download_model = gr.Button('5.Download Model')
                    with gr.Row():
                        model_files = gr.Files(label='Your Model and Index file can be downloaded here:')
                        download_model.click(
                            fn=lambda name: os.listdir(f'assets/weights/{name}') + glob.glob(f'logs/{name.split(".")[0]}/added_*.index'),
                            inputs=[training_name], 
                            outputs=[model_files, info3])
                    with gr.Row():
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
                            [f0method8, pretrained_G14, pretrained_D15],
                        )
                    

    if config.iscolab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )

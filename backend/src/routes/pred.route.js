import { Router } from "express";
import path from "path";
import { spawn } from "child_process";
import fs from "fs";
import dotenv from "dotenv";
import formidable from "formidable";

// 加载环境变量
dotenv.config();

const router = Router();

// 从环境变量中读取 Python 路径
const PYTHON_PATH = process.env.PYTHON_PATH || "E:\\Anaconda\\envs\\MER\\python.exe";

// 验证 PYTHON_PATH 是否存在
if (!PYTHON_PATH) {
    throw new Error("PYTHON_PATH 环境变量未设置");
}

// 定义音频和 JSON 保存路径
const AUDIO_SAVE_PATH = path.join(__dirname, "../../frontend/public/songs");
const JSON_SAVE_PATH = path.join(__dirname, "../../frontend/public");

// POST /api/process-audio
router.post("/", (req, res) => {
    const form = new formidable.IncomingForm();
    form.uploadDir = AUDIO_SAVE_PATH;
    form.keepExtensions = true;

    form.parse(req, (err, fields, files) => {
        if (err) {
            return res.status(500).json({ error: err.message });
        }

        const audioFile = files.audioFile;
        if (!audioFile) {
            return res.status(400).json({ error: "请上传音频文件" });
        }

        const audioPath = audioFile.filepath || audioFile.path;

        // 定义 Python 脚本路径
        const pythonScriptPath = path.join(__dirname, "../pyuntil/process_audio.py");

        // 调用 Python 脚本
        const pythonProcess = spawn(PYTHON_PATH, [pythonScriptPath, audioPath]);

        let jsonOutput = "";
        let errorOutput = "";

        pythonProcess.stdout.on("data", (data) => {
            jsonOutput += data.toString();
        });

        pythonProcess.stderr.on("data", (data) => {
            errorOutput += data.toString();
        });

        pythonProcess.on("close", (code) => {
            if (code !== 0) {
                console.error(`Python 脚本退出，退出码 ${code}`);
                console.error(`错误信息: ${errorOutput}`);
                return res.status(500).json({ error: "音频处理时出错" });
            }

            try {
                const jsonData = JSON.parse(jsonOutput);

                // 生成 JSON 文件名和路径
                const jsonFilename = `${path.basename(audioFile.originalFilename || audioFile.newFilename, path.extname(audioFile.originalFilename || audioFile.newFilename))}.json`;
                const jsonFilePath = path.join(JSON_SAVE_PATH, jsonFilename);

                // 保存 JSON 文件
                fs.writeFileSync(jsonFilePath, JSON.stringify(jsonData, null, 2), "utf-8");

                res.json({ message: "音频处理成功", jsonData: jsonData });
            } catch (err) {
                console.error("解析 JSON 时出错:", err);
                res.status(500).json({ error: "生成的 JSON 数据无效" });
            }
        });
    });
});

export default router;
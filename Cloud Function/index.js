/**
 * yoga-platform 云函数（优化版·2025-05-22）
 * -------------------------------------------------------------
 *  • action === 'loadPoseSequence' → 返回指定等级训练序列 JSON 的
 *    COS 私有签名 URL（1 h 有效）
 *  • 其它 / 无 action              → 下载 zip ➜ 解压 ➜ 返回文件统计
 *    （不返回文件列表，避免响应过大）
 * -------------------------------------------------------------
 */

const COS      = require('cos-nodejs-sdk-v5');
const unzipper = require('unzipper');
const fs       = require('fs');
const path     = require('path');

/* ------------ 0. 运行时凭据 & 常量 ------------ */
// 临时密钥由 TC 函数运行时自动注入，更安全
const cos = new COS({
  SecretId:        process.env.TENCENTCLOUD_SECRETID,
  SecretKey:       process.env.TENCENTCLOUD_SECRETKEY,
  XCosSecurityToken: process.env.TENCENTCLOUD_SESSIONTOKEN,
});

// ⚠️ 如需多环境共用，请改为读环境变量
const Bucket    = 'yogasmart-static-1351554677';    // 不含 appId
const Region    = 'ap-shanghai';
const ObjectKey = 'static/deps/yoga-platform.zip';  // 旧 ZIP 包
const ZIP_PATH  = '/tmp/yoga-platform.zip';
const UNZIP_DIR = '/tmp/yoga-platform';

/* ------------ 1. COS 私有对象签名 URL ------------ */
/**
 * 生成 COS 对象的签名 URL
 * @param {string} Key - COS 对象键名
 * @param {number} expires - 签名有效期（秒），默认 1 小时
 * @returns {Promise<string>} 签名 URL
 */
function getSignedUrl(Key, expires = 3600) {
  return new Promise((resolve, reject) => {
    cos.getObjectUrl({ 
      Bucket, 
      Region, 
      Key, 
      Sign: true, 
      Expires: expires 
    }, (err, data) => {
      if (err) {
        console.error(`[getSignedUrl] 获取签名 URL 失败:`, err);
        reject(err);
      } else {
        console.log(`[getSignedUrl] 成功生成签名 URL: ${Key}`);
        resolve(data.Url);
      }
    });
  });
}

/* ------------ 2. 下载 & 解压工具函数 ------------ */
/**
 * 从 COS 下载文件到本地
 * @param {string} Key - COS 对象键名
 * @param {string} filePath - 本地文件路径
 * @returns {Promise<void>}
 */
function downloadFromCOS(Key, filePath) {
  return new Promise((resolve, reject) => {
    try {
      // 清理可能存在的旧文件
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
        console.log(`[downloadFromCOS] 清理旧文件: ${filePath}`);
      }
      
      const out = fs.createWriteStream(filePath);
      
      cos.getObject({ 
        Bucket, 
        Region, 
        Key, 
        Output: out 
      }, (err) => {
        if (err) {
          console.error(`[downloadFromCOS] 下载失败:`, err);
          reject(err);
        } else {
          console.log(`[downloadFromCOS] 下载成功: ${Key} → ${filePath}`);
          resolve();
        }
      });
    } catch (error) {
      console.error(`[downloadFromCOS] 预处理失败:`, error);
      reject(error);
    }
  });
}

/**
 * 解压 ZIP 文件到指定目录
 * @param {string} zipPath - ZIP 文件路径
 * @param {string} destDir - 目标解压目录
 * @returns {Promise<void>}
 */
function unzipFile(zipPath, destDir) {
  return new Promise((resolve, reject) => {
    try {
      // 确保目标目录存在
      if (!fs.existsSync(destDir)) {
        fs.mkdirSync(destDir, { recursive: true });
        console.log(`[unzipFile] 创建目录: ${destDir}`);
      }
      
      fs.createReadStream(zipPath)
        .pipe(unzipper.Extract({ path: destDir }))
        .on('close', () => {
          console.log(`[unzipFile] 解压完成: ${zipPath} → ${destDir}`);
          resolve();
        })
        .on('error', (err) => {
          console.error(`[unzipFile] 解压失败:`, err);
          reject(err);
        });
    } catch (error) {
      console.error(`[unzipFile] 预处理失败:`, error);
      reject(error);
    }
  });
}

/**
 * 递归统计目录中的文件数量（不返回文件列表，避免响应过大）
 * @param {string} dir - 目录路径
 * @returns {number} 文件总数
 */
function countFilesRecursive(dir) {
  let fileCount = 0;
  
  try {
    if (!fs.existsSync(dir)) {
      console.warn(`[countFilesRecursive] 目录不存在: ${dir}`);
      return 0;
    }
    
    const items = fs.readdirSync(dir);
    
    for (const item of items) {
      const fullPath = path.join(dir, item);
      const stat = fs.statSync(fullPath);
      
      if (stat.isDirectory()) {
        // 递归统计子目录
        fileCount += countFilesRecursive(fullPath);
      } else if (stat.isFile()) {
        // 统计文件
        fileCount++;
      }
    }
  } catch (error) {
    console.error(`[countFilesRecursive] 统计文件失败:`, error);
  }
  
  return fileCount;
}

/**
 * 获取 /tmp 目录剩余空间（MB）
 * @returns {string} 剩余空间大小（MB），失败时返回空字符串
 */
function getTmpFreeSpaceMB() {
  try {
    const { execSync } = require('child_process');
    const output = execSync('df -k /tmp', { encoding: 'utf8' });
    const lines = output.trim().split('\n');
    
    if (lines.length >= 2) {
      const [, , , freeKB] = lines[1].trim().split(/\s+/);
      const freeMB = (parseInt(freeKB, 10) / 1024).toFixed(2);
      return freeMB;
    }
    
    return 'unknown';
  } catch (error) {
    console.warn(`[getTmpFreeSpaceMB] 获取磁盘空间失败:`, error);
    return 'unknown';
  }
}

/**
 * 清理临时文件
 * @param {string[]} paths - 要清理的文件/目录路径
 */
function cleanupTempFiles(paths) {
  for (const filePath of paths) {
    try {
      if (fs.existsSync(filePath)) {
        const stat = fs.statSync(filePath);
        if (stat.isDirectory()) {
          fs.rmSync(filePath, { recursive: true, force: true });
        } else {
          fs.unlinkSync(filePath);
        }
        console.log(`[cleanup] 清理: ${filePath}`);
      }
    } catch (error) {
      console.warn(`[cleanup] 清理失败 ${filePath}:`, error.message);
    }
  }
}

/* ------------ 3. 云函数主入口 ------------ */
/**
 * 云函数主入口
 * @param {Object} event - 事件参数
 * @param {string} event.action - 操作类型
 * @param {string} event.level - 难度等级（默认 beginner）
 * @param {Object} context - 上下文信息
 * @returns {Object} 响应结果
 */
exports.main = async (event = {}, context) => {
  const startTime = Date.now();
  const { action, level = 'beginner' } = event;
  
  console.log(`[main] 开始处理请求 - action: ${action}, level: ${level}`);
  console.log(`[main] /tmp 初始空间: ~${getTmpFreeSpaceMB()} MB`);

  /* 3-1 仅返回训练序列 JSON URL（≈ 1 KB，极快） */
  if (action === 'loadPoseSequence') {
    const jsonKey = `static/pose-sequences/${level}.json`;
    
    try {
      console.log(`[loadPoseSequence] 开始生成签名 URL: ${jsonKey}`);
      const url = await getSignedUrl(jsonKey);
      
      const duration = Date.now() - startTime;
      console.log(`[loadPoseSequence] 成功完成，耗时: ${duration}ms`);
      
      return { 
        code: 0, 
        url,
        message: `成功获取 ${level} 级别训练序列`,
        timestamp: new Date().toISOString(),
        duration: `${duration}ms`
      };
    } catch (error) {
      const duration = Date.now() - startTime;
      console.error('[loadPoseSequence] 失败:', error);
      
      return { 
        code: 1, 
        message: '获取训练序列失败', 
        error: error.message || error.toString(),
        level,
        timestamp: new Date().toISOString(),
        duration: `${duration}ms`
      };
    }
  }

  /* 3-2 默认流程：下载 zip ➜ 解压 ➜ 返回统计信息（不返回文件列表） */
  try {
    console.log('[ZIP] 开始 ZIP 处理流程');
    console.log('[ZIP] 开始从 COS 下载 ZIP 文件...');

    // 步骤1：下载 ZIP 文件
    await downloadFromCOS(ObjectKey, ZIP_PATH);
    
    // 检查文件大小
    const zipStats = fs.statSync(ZIP_PATH);
    console.log(`[ZIP] ZIP 文件大小: ${(zipStats.size / 1024 / 1024).toFixed(2)} MB`);

    console.log('[ZIP] 下载完成，开始解压...');
    
    // 步骤2：解压文件
    await unzipFile(ZIP_PATH, UNZIP_DIR);

    console.log('[ZIP] 解压完成，开始统计文件...');
    
    // 步骤3：统计文件数量（不生成文件列表，避免响应过大）
    const fileCount = countFilesRecursive(UNZIP_DIR);
    
    const duration = Date.now() - startTime;
    const finalFreeMB = getTmpFreeSpaceMB();
    
    console.log(`[ZIP] 处理完成 - 文件数: ${fileCount}, 耗时: ${duration}ms`);

    // 清理临时文件（可选，云函数容器回收时会自动清理）
    setTimeout(() => {
      cleanupTempFiles([ZIP_PATH, UNZIP_DIR]);
    }, 1000);

    return {
      code: 0,
      message: '成功下载并解压 ZIP 文件',
      fileCount,
      zipSizeMB: (zipStats.size / 1024 / 1024).toFixed(2),
      tmpFreeMB: finalFreeMB,
      timestamp: new Date().toISOString(),
      duration: `${duration}ms`
    };

  } catch (error) {
    const duration = Date.now() - startTime;
    console.error('[ZIP] 流程出错:', error);

    // 发生错误时也清理临时文件
    cleanupTempFiles([ZIP_PATH, UNZIP_DIR]);

    return {
      code: 1,
      message: 'ZIP 处理失败: ' + (error.message || error.toString()),
      error: error.stack || error.toString(),
      tmpFreeMB: getTmpFreeSpaceMB(),
      timestamp: new Date().toISOString(),
      duration: `${duration}ms`
    };
  }
};
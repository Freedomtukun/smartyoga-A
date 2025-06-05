/*----------------------------------------------------
 * utils/yoga-api.js  ◇ SmartYoga Mini-Program
 * - Handles uploading user videos for pose scoring.
 * - Provides URLs for resulting skeleton images.
 * - Downloads skeleton images to local temporary paths.
 * - Enhanced error handling for NO_KEYPOINT responses
 *--------------------------------------------------*/

/** 后端 API 根域名 (HTTPS, whitelisted in WeChat admin panel) */
export const API_BASE_URL = 'https://api.yogasmart.cn';

/** Backend route for pose scoring and skeleton image generation. */
export const DETECT_POSE_URL = API_BASE_URL + '/api/detect-pose-file';

/** Placeholder image for failed pose image loading */
export const DEFAULT_POSE_IMAGE = '/assets/images/placeholder.png';

/** Common headers for API requests */
const COMMON_HEADERS = {
  'Accept': 'application/json'
};

/* ---------- Utility Functions ---------- */

/**
 * Safely parses a JSON string.
 * @param {string} str - The string to parse.
 * @returns {object|null} The parsed object or null if parsing fails.
 */
function safeJSONParse(str) {
  try { 
    return JSON.parse(str); 
  } catch (e) { 
    console.error('[safeJSONParse] Parse error:', e, 'Original string:', str);
    return null; 
  }
}

/**
 * Uploads a video/image file and gets the pose scoring results.
 * Returns an object containing both the promise and the upload task.
 * Enhanced with NO_KEYPOINT handling
 *
 * @param {string} filePath - The temporary path of the video/image file
 * @param {string} poseId - The ID of the current pose
 * @returns {{promise: Promise<object>, task: wx.UploadTask}} 
 *          An object containing the promise and upload task
 */
export function uploadAndScore(filePath, poseId) {
  console.log('[uploadAndScore] Starting upload to:', DETECT_POSE_URL, 'with poseId:', poseId);
  
  // New debug log for easier debugging
  console.log(`[UPLOAD] POST → ${DETECT_POSE_URL}`);

  let uploadTask = null;

  const promise = new Promise((resolve, reject) => {
    uploadTask = wx.uploadFile({
      url: DETECT_POSE_URL,
      filePath,
      name: 'file',
      header: COMMON_HEADERS,
      formData: { poseId },
      timeout: 60000, // 60s timeout
      success: (res) => {
        const { statusCode, data } = res;
        console.log('[uploadAndScore] Response received:', {
          url: DETECT_POSE_URL,
          statusCode,
          dataLength: data ? data.length : 0
        });

        // 现在所有响应都应该是 200
        if (statusCode !== 200) {
          console.error('[uploadAndScore] HTTP Error', {
            url: DETECT_POSE_URL,
            statusCode,
            responseData: data,
            poseId,
            filePath
          });
          
          // 即使是错误状态码，也尝试解析响应
          const json = safeJSONParse(data);
          if (json && json.code) {
            // 如果有结构化的错误响应，使用它
            resolve({
              score: json.score || 0,
              skeletonUrl: json.skeletonUrl || null,
              code: json.code,
              msg: json.msg || `HTTP ${statusCode}`,
              poseId: json.poseId || poseId
            });
            return;
          }
          
          // 否则返回默认错误响应
          resolve({
            score: 0,
            skeletonUrl: null,
            code: "HTTP_ERROR",
            msg: `HTTP ${statusCode} — ${DETECT_POSE_URL}`,
            poseId: poseId
          });
          return;
        }

        const json = safeJSONParse(data);
        if (!json) {
          console.error('[uploadAndScore] Invalid JSON response', {
            url: DETECT_POSE_URL,
            statusCode,
            rawData: data,
            poseId,
            filePath
          });
          
          // 返回错误响应而不是 reject
          resolve({
            score: 0,
            skeletonUrl: null,
            code: "INVALID_JSON",
            msg: "Invalid JSON response",
            poseId: poseId
          });
          return;
        }
        
        // 处理 NO_KEYPOINT 响应
        if (json.code === "NO_KEYPOINT") {
          console.log('[uploadAndScore] No keypoints detected:', {
            poseId: json.poseId || poseId,
            score: 0,
            msg: json.msg
          });
          
          // 返回结构化的响应，分数为0
          resolve({
            score: 0,
            skeletonUrl: null,
            keypoints: [],
            label: 'unknown',
            suggestion: json.msg || "无法检测到姿势关键点",
            code: "NO_KEYPOINT",
            poseId: json.poseId || poseId
          });
          return;
        }
        
        // 处理其他错误代码
        if (json.code === "ERROR" || json.code === "NO_FILE") {
          console.log('[uploadAndScore] Server error:', {
            code: json.code,
            msg: json.msg,
            poseId: json.poseId || poseId
          });
          
          resolve({
            score: json.score || 0,
            skeletonUrl: json.skeletonUrl || null,
            code: json.code,
            msg: json.msg,
            poseId: json.poseId || poseId
          });
          return;
        }

        // 正常成功响应
        console.log('[uploadAndScore] Success:', {
          url: DETECT_POSE_URL,
          poseId: json.poseId || poseId,
          score: json.score,
          skeletonUrl: json.skeletonUrl,
          code: json.code || "SUCCESS"
        });
        
        resolve({
          ...json,
          code: json.code || "SUCCESS",
          poseId: json.poseId || poseId
        });
      },
      fail: (err) => {
        console.error('[uploadAndScore] wx.uploadFile failed:', {
          url: DETECT_POSE_URL,
          poseId,
          filePath,
          error: err,
          errMsg: err.errMsg,
          errCode: err.errCode
        });
        
        // Check if it was user-initiated cancellation
        const wasAborted = err.errMsg && err.errMsg.includes('abort');
        
        // 返回结构化的错误响应而不是 reject
        resolve({
          score: 0,
          skeletonUrl: null,
          code: wasAborted ? "ABORTED" : "UPLOAD_FAILED",
          msg: wasAborted ? 'Upload cancelled by user' : `Upload failed: ${err.errMsg || 'Unknown error'}`,
          wasAborted,
          poseId: poseId
        });
      }
    });

    // Progress monitoring
    uploadTask.onProgressUpdate((res) => {
      console.log('[uploadAndScore] Upload progress:', res.progress + '%');
    });
  });

  return { promise, task: uploadTask };
}

/**
 * Wrapper function for frame scoring that matches the expected interface
 * Enhanced with NO_KEYPOINT handling
 * @param {string} framePath - The temporary path of the frame image
 * @param {string} poseId - The ID of the current pose
 * @returns {{promise: Promise<object>, task: wx.UploadTask}}
 */
export function uploadFrameForScoring(framePath, poseId) {
  const result = uploadAndScore(framePath, poseId);
  
  // 包装 promise 以确保总是返回有效的分数
  const enhancedPromise = result.promise.then(data => {
    // 如果是 NO_KEYPOINT 或其他错误，确保有默认值
    if (data.code === "NO_KEYPOINT" || data.code === "ERROR") {
      return {
        score: 0,
        skeletonUrl: data.skeletonUrl || null,
        keypoints: data.keypoints || [],
        label: data.label || 'unknown',
        suggestion: data.suggestion || data.msg || "检测失败",
        code: data.code,
        poseId: data.poseId
      };
    }
    
    // 确保所有字段都有默认值
    return {
      score: data.score || 0,
      skeletonUrl: data.skeletonUrl || null,
      keypoints: data.keypoints || [],
      label: data.label || 'unknown',
      suggestion: data.suggestion || '',
      code: data.code || "SUCCESS",
      poseId: data.poseId
    };
  });
  
  return {
    promise: enhancedPromise,
    task: result.task
  };
}

/**
 * Downloads a skeleton image to a local temporary path.
 * @param {string} url - The public URL of the skeleton image
 * @return {Promise<string>} A promise that resolves to the temporary file path
 */
export function downloadSkeletonImage(url) {
  console.log('[downloadSkeletonImage] Starting download from:', url);
  
  return new Promise((resolve, reject) => {
    // 如果 URL 为空，直接返回默认图片
    if (!url) {
      console.log('[downloadSkeletonImage] No URL provided, using default image');
      resolve(DEFAULT_POSE_IMAGE);
      return;
    }
    
    wx.downloadFile({
      url,
      success: (res) => {
        const { statusCode, tempFilePath } = res;
        console.log('[downloadSkeletonImage] Response:', {
          url,
          statusCode,
          tempFilePath
        });
        
        if (statusCode === 200) {
          resolve(tempFilePath);
        } else {
          console.error('[downloadSkeletonImage] HTTP Error:', {
            url,
            statusCode,
            tempFilePath
          });
          // 下载失败时返回默认图片而不是 reject
          resolve(DEFAULT_POSE_IMAGE);
        }
      },
      fail: (err) => {
        console.error('[downloadSkeletonImage] wx.downloadFile failed:', {
          url,
          error: err,
          errMsg: err.errMsg
        });
        // 下载失败时返回默认图片而不是 reject
        resolve(DEFAULT_POSE_IMAGE);
      }
    });
  });
}

/**
 * 批量处理帧评分结果
 * 过滤掉检测失败的帧，计算平均分
 * @param {Array} scoreResults - 评分结果数组
 * @returns {Object} 处理后的结果
 */
export function processFrameScores(scoreResults) {
  // 过滤出有效的分数（排除 NO_KEYPOINT 等错误）
  const validScores = scoreResults.filter(result => {
    return result.code === "SUCCESS" && result.score > 0;
  });
  
  // 记录统计信息
  const stats = {
    total: scoreResults.length,
    valid: validScores.length,
    failed: scoreResults.length - validScores.length,
    noKeypoint: scoreResults.filter(r => r.code === "NO_KEYPOINT").length
  };
  
  console.log('[processFrameScores] Statistics:', stats);
  
  // 如果没有有效分数，返回0
  if (validScores.length === 0) {
    return {
      averageScore: 0,
      validFrames: 0,
      totalFrames: scoreResults.length,
      stats,
      message: "未能检测到有效的姿势"
    };
  }
  
  // 计算平均分
  const totalScore = validScores.reduce((sum, result) => sum + result.score, 0);
  const averageScore = Math.round(totalScore / validScores.length);
  
  return {
    averageScore,
    validFrames: validScores.length,
    totalFrames: scoreResults.length,
    stats,
    validScores,
    message: `成功分析 ${validScores.length}/${scoreResults.length} 帧`
  };
}

/* ------------------------------------------------------------------
 * Compatibility for older code: 'scorePose' is an alias for 'uploadAndScore'.
 * -----------------------------------------------------------------*/
export const scorePose = uploadAndScore;
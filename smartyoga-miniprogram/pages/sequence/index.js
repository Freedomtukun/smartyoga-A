import { uploadFrameForScoring, DEFAULT_POSE_IMAGE } from '../../utils/yoga-api.js';
const cloudSequenceService = require('../../utils/cloud-sequence-service.js');
const sequenceService = require('../../utils/sequence-service.js');
const getText = v => (typeof v === 'object' ? (v.zh || v.en || '') : v);

Page({
  data: {
    level: '',
    currentSequence: null,
    currentPoseIndex: 0,
    isPlaying: false,
    timeRemaining: 0,
    loading: true,
    error: null,
    skeletonUrl: null,
    timerId: null,
    recordedVideo: null,
    isProcessingFrames: false,
    frameAnalysisResults: [],
    topThreeFrames: [],
    isCancelling: false,
    currentUploadTasks: [],
    failedUploads: [],
    videoMetadata: {
      duration: 0,
      width: 0,
      height: 0
    },
    frameExtractionCanvasContext: null,
    frameExtractorVideoContext: null,
    extractorVideoSrc: null,
    defaultPoseImage: DEFAULT_POSE_IMAGE
  },

  // Initialize canvas and video contexts for frame extraction
  initializeFrameExtractionResources: function() {
    if (!this.data.frameExtractionCanvasContext) {
      const ctx = wx.createCanvasContext('frameExtractorCanvas', this);
      if (!ctx) {
        console.error('[INIT] Failed to create canvas context "frameExtractorCanvas"');
      }
      this.setData({ frameExtractionCanvasContext: ctx });
    }
    if (!this.data.frameExtractorVideoContext) {
      const videoCtx = wx.createVideoContext('frameExtractorVideo', this);
      if (!videoCtx) {
        console.error('[INIT] Failed to create video context "frameExtractorVideo"');
      }
      this.setData({ frameExtractorVideoContext: videoCtx });
    }
  },

  // Called when the hidden video element has loaded its metadata
  onVideoLoadMetadata: function(e) {
    wx.hideLoading();

    const { duration, width, height } = e.detail;
    console.log('[VIDEO_META] Loaded:', { duration, width, height });

    if (!duration || duration <= 0 || !width || width <= 0 || !height || height <= 0) {
      console.error('[VIDEO_META] Invalid metadata:', { duration, width, height });
      this.setData({ isProcessingFrames: false });

      wx.showModal({
        title: '提示',
        content: '当前环境或视频格式不支持，请在真机上重试。',
        showCancel: false,
        confirmText: '知道了'
      });
      return;
    }

    this.setData({ 
      videoMetadata: { duration, width, height }
    });
    
    this.startFrameExtractionLoop();
  },

  // Core logic for extracting frames from the video
  startFrameExtractionLoop: async function() {
    console.log('[FRAME_EXTRACTION] Starting frame extraction');
    
    this.setData({
      isProcessingFrames: true,
      frameAnalysisResults: [],
      topThreeFrames: [],
      isCancelling: false
    });

    const { duration } = this.data.videoMetadata;
    const videoCtx = this.data.frameExtractorVideoContext;
    const canvasCtx = this.data.frameExtractionCanvasContext;

    if (!videoCtx || !canvasCtx) {
      console.error('[FRAME_EXTRACTION] Missing contexts');
      this.setData({ isProcessingFrames: false });
      wx.showToast({ title: '资源错误', icon: 'none' });
      return;
    }

    const originalWidth = this.data.videoMetadata.width;
    const originalHeight = this.data.videoMetadata.height;
    let targetWidth = originalWidth;
    let targetHeight = originalHeight;

    if (originalWidth > 480) {
      targetWidth = 480;
      targetHeight = Math.round(originalHeight * (480 / originalWidth));
    }
    console.log(`[FRAME_EXTRACTION] Target dimensions: ${targetWidth}x${targetHeight}`);

    const extractedFramePaths = [];

    // Extract frames at 2-second intervals
    for (let t = 0; t < duration; t += 2) {
      if (this.data.isCancelling) {
        console.log('[FRAME_EXTRACTION] Cancelled by user');
        this.setData({ isProcessingFrames: false, isCancelling: false });
        wx.hideLoading();
        return;
      }

      videoCtx.seek(t);
      await new Promise(resolve => setTimeout(resolve, 500));

      if (this.data.isCancelling) break;

      canvasCtx.drawImage('frameExtractorVideo', 0, 0, targetWidth, targetHeight);
      await new Promise(resolve => canvasCtx.draw(false, resolve));

      if (this.data.isCancelling) break;

      try {
        const frameData = await wx.canvasToTempFilePath({
          x: 0, y: 0,
          width: targetWidth, height: targetHeight,
          destWidth: targetWidth, destHeight: targetHeight,
          canvasId: 'frameExtractorCanvas',
          fileType: 'jpg', 
          quality: 0.7
        }, this);
        extractedFramePaths.push(frameData.tempFilePath);
        console.log(`[FRAME_EXTRACTION] Frame at ${t}s saved:`, frameData.tempFilePath);
      } catch (err) {
        console.error(`[FRAME_EXTRACTION] Failed at ${t}s:`, err);
      }
    }

    if (this.data.isCancelling) {
      console.log('[FRAME_EXTRACTION] Final cancellation check');
      this.setData({ isProcessingFrames: false, isCancelling: false });
      wx.hideLoading();
      return;
    }

    console.log('[FRAME_EXTRACTION] Completed. Total frames:', extractedFramePaths.length);

    if (extractedFramePaths.length > 0) {
      this.analyzeFramesBatch(extractedFramePaths);
    } else {
      this.setData({ isProcessingFrames: false });
      wx.showToast({ title: '未能成功提取任何帧', icon: 'none' });
    }
  },

  // Handles user-initiated cancellation
  handleCancelUpload: function() {
    console.log('[CANCEL] User initiated cancellation');
    this.setData({ isCancelling: true });

    // Abort all ongoing upload tasks
    const tasks = this.data.currentUploadTasks;
    if (tasks && tasks.length > 0) {
      console.log(`[CANCEL] Aborting ${tasks.length} upload tasks`);
      tasks.forEach(task => {
        if (task && typeof task.abort === 'function') {
          task.abort();
        }
      });
    }

    this.setData({
      isProcessingFrames: false,
      currentUploadTasks: [],
      frameAnalysisResults: [],
      topThreeFrames: []
    });
    
    wx.hideLoading();
    wx.showToast({ title: '已取消处理', icon: 'none' });
  },

  // Analyzes a batch of extracted frames
  analyzeFramesBatch: async function(framePathsArray, _poseId = null) {
    if (this.data.isCancelling) {
      console.log('[ANALYZE_BATCH] Skipped due to cancellation');
      this.setData({ isProcessingFrames: false, isCancelling: false });
      wx.hideLoading();
      return;
    }

    if (!framePathsArray || framePathsArray.length === 0) {
      if (!_poseId) {
        wx.showToast({ title: '没有提取到帧进行分析', icon: 'none' });
      }
      this.setData({ isProcessingFrames: false });
      return;
    }

    this.setData({ isProcessingFrames: true, currentUploadTasks: [] });
    wx.showLoading({ title: '分析中 (0%)...', mask: true });

    const poseId = _poseId || (
      this.data.currentSequence && 
      this.data.currentSequence.poses[this.data.currentPoseIndex] && 
      this.data.currentSequence.poses[this.data.currentPoseIndex].id
    );

    if (!poseId) {
      console.error('[ANALYZE_BATCH] No pose ID found');
      this.setData({ isProcessingFrames: false, isCancelling: false, currentUploadTasks: [] });
      wx.hideLoading();
      wx.showToast({ title: '无法确定体式ID', icon: 'none' });
      return;
    }

    console.log('[ANALYZE_BATCH] Starting analysis for poseId:', poseId);

    const BATCH_SIZE = 3;
    const totalFrames = framePathsArray.length;
    let processedCount = 0;
    let successfulUploads = 0;

    if (!_poseId) {
      this.setData({ frameAnalysisResults: [] });
    }

    for (let i = 0; i < totalFrames; i += BATCH_SIZE) {
      if (this.data.isCancelling) {
        console.log('[ANALYZE_BATCH] Cancelled in main loop');
        break;
      }

      const currentBatchPaths = framePathsArray.slice(i, i + BATCH_SIZE);
      const uploadPromises = [];
      const batchTasks = [];

      for (const framePath of currentBatchPaths) {
        if (this.data.isCancelling) break;

        processedCount++;
        wx.showLoading({ 
          title: `分析第 ${processedCount}/${totalFrames} 帧...`, 
          mask: true 
        });
        
        const { promise, task } = uploadFrameForScoring(framePath, poseId);
        
        // Process the promise to adapt the result structure
        const adaptedPromise = promise
          .then(result => {
            console.log('[UPLOAD_FRAME] Success:', result);
            return {
              score: result.score || 0,
              feedback: result.feedback || "评分完成",
              skeletonUrl: result.skeletonUrl,
              originalFramePath: framePath
            };
          })
          .catch(err => {
            console.error('[UPLOAD_FRAME] Failed:', err);
            const wasAborted = err.wasAborted || false;
            const errorMessage = err.message || 'Upload failed';
            
            return Promise.reject({
              error: errorMessage,
              originalFramePath: framePath,
              details: err,
              wasAborted: wasAborted
            });
          });
          
        uploadPromises.push(adaptedPromise);
        if (task) {
          batchTasks.push(task);
        }
      }
      
      // Add tasks to global list
      if (batchTasks.length > 0) {
        const currentTasks = this.data.currentUploadTasks;
        this.setData({ 
          currentUploadTasks: [...currentTasks, ...batchTasks] 
        });
      }

      if (this.data.isCancelling) break;

      const batchResults = await Promise.allSettled(uploadPromises);
      
      // Remove completed tasks from global list
      if (batchTasks.length > 0) {
        const updatedTasks = this.data.currentUploadTasks.filter(t => !batchTasks.includes(t));
        this.setData({ currentUploadTasks: updatedTasks });
      }

      if (this.data.isCancelling) break;
      
      const currentResults = [...this.data.frameAnalysisResults];
      batchResults.forEach(result => {
        if (result.status === 'fulfilled') {
          currentResults.push(result.value);
          successfulUploads++;
        } else {
          currentResults.push(result.reason);
        }
      });
      this.setData({ frameAnalysisResults: currentResults });
    }

    wx.hideLoading();
    
    if (this.data.isCancelling) {
      console.log('[ANALYZE_BATCH] Process was cancelled');
      this.setData({ isProcessingFrames: false, isCancelling: false, currentUploadTasks: [] });
      return;
    }

    this.setData({ currentUploadTasks: [] });

    // Store failed uploads info
    const sessionFailedUploads = this.data.frameAnalysisResults
      .filter(r => r.error && !r.wasAborted)
      .map(r => ({ 
        framePath: r.originalFramePath, 
        poseId: poseId, 
        error: r.error 
      }));
    this.setData({ failedUploads: sessionFailedUploads });

    if (sessionFailedUploads.length > 0) {
      wx.showToast({ 
        title: `${sessionFailedUploads.length} 帧上传失败`, 
        icon: 'none', 
        duration: 3000 
      });
    }

    console.log('[ANALYZE_BATCH] Complete. Success:', successfulUploads, 'Failed:', sessionFailedUploads.length);
    
    if (successfulUploads === 0 && totalFrames > 0 && !this.data.isCancelling) {
      wx.showToast({ title: '所有帧分析失败', icon: 'none', duration: 2000 });
    } else if (successfulUploads === totalFrames && totalFrames > 0) {
      console.log('[ANALYZE_BATCH] All frames analyzed successfully');
    }
    
    this.selectAndDisplayTopFrames();
    this.setData({ isCancelling: false });
  },

  // Retry failed uploads
  handleRetryFailedUploads: function() {
    if (!this.data.failedUploads || this.data.failedUploads.length === 0) {
      wx.showToast({ title: '没有失败的上传', icon: 'none' });
      return;
    }

    console.log('[RETRY] Starting retry for', this.data.failedUploads.length, 'failed uploads');

    const framesToRetryInfo = [...this.data.failedUploads];
    const framesToRetryPaths = framesToRetryInfo.map(f => f.framePath);
    const poseIdForRetry = framesToRetryInfo[0]?.poseId;

    this.setData({ failedUploads: [] });

    if (framesToRetryPaths.length > 0 && poseIdForRetry) {
      this.setData({ isProcessingFrames: true });

      // Filter out previous failed attempts
      const currentResults = this.data.frameAnalysisResults.filter(
        r => !framesToRetryPaths.includes(r.originalFramePath) || (r.originalFramePath && r.score > 0)
      );
      this.setData({ frameAnalysisResults: currentResults });

      this.analyzeFramesBatch(framesToRetryPaths, poseIdForRetry);
    } else {
      wx.showToast({ title: '无法重试', icon: 'none' });
      this.setData({ isProcessingFrames: false });
    }
  },

  // Select and display top 3 frames
  selectAndDisplayTopFrames: function() {
    if (this.data.isCancelling) {
      this.setData({ isProcessingFrames: false, isCancelling: false, topThreeFrames: [] });
      return;
    }

    const results = this.data.frameAnalysisResults;
    this.setData({ isProcessingFrames: false });

    if (!results || results.length === 0) {
      this.setData({ topThreeFrames: [] });
      return;
    }

    const validResults = results.filter(r => 
      r && typeof r.score === 'number' && r.score > 0 && r.skeletonUrl && !r.wasAborted
    );
    
    if (validResults.length === 0) {
      this.setData({ topThreeFrames: [] });
      if (results.filter(r => !r.wasAborted).length > 0) {
        wx.showToast({ title: '未选出足够评分的帧', icon: 'none' });
      }
      return;
    }

    validResults.sort((a, b) => b.score - a.score);
    const topFrames = validResults.slice(0, 3);

    this.setData({ topThreeFrames: topFrames });
    console.log('[TOP_FRAMES] Selected top 3 frames:', topFrames);

    if (topFrames.length > 0) {
      wx.showToast({ 
        title: `最佳 ${topFrames.length} 帧已显示`, 
        icon: 'success', 
        duration: 2000 
      });
    }
    this.setData({ isCancelling: false });
  },

  // Process video for frame extraction
  processVideoForFrames: function(videoPath) {
    console.log('[PROCESS_VIDEO] Starting processing for:', videoPath);
    
    this.setData({ 
      isProcessingFrames: true, 
      topThreeFrames: [], 
      frameAnalysisResults: [],
      isCancelling: false, 
      currentUploadTasks: []
    });
    
    wx.showLoading({ title: '准备视频分析...', mask: true });
    
    this.initializeFrameExtractionResources();
    this.setData({ extractorVideoSrc: videoPath });
  },

  // Main upload and score function
  async uploadAndScore() {
    if (!this.data.recordedVideo) {
      wx.showToast({ title: '请先录制视频', icon: 'none' });
      return;
    }

    console.log('[MAIN] Starting upload and score process');

    this.setData({
      isProcessingFrames: true,
      topThreeFrames: [],
      frameAnalysisResults: [],
      isCancelling: false,
      currentUploadTasks: [],
      failedUploads: []
    });

    wx.showLoading({ title: '处理准备中...', mask: true });

    try {
      await this.processVideoForFrames(this.data.recordedVideo);
    } catch (error) {
      console.error('[MAIN] Error starting video processing:', error);
      this.setData({ isProcessingFrames: false, isCancelling: false });
      wx.hideLoading();
      wx.showToast({ title: '处理启动失败', icon: 'none' });
    }
  },

  // Page lifecycle: Load sequence data
  onLoad: function (options) {
    const level = options.level || 'beginner';
    this.setData({ level: level });
    this.loadSequenceData(level);
  },

  // Load sequence data
  async loadSequenceData(level) {
    console.log('[LOAD] Loading sequence for level:', level);
    this.setData({ loading: true, error: null });
    wx.showLoading({ title: '加载中...' });
    
    try {
      const sequenceData = await cloudSequenceService.getProcessedSequence(level);
      
      if (sequenceData && sequenceData.poses && sequenceData.poses.length > 0) {
        const initialState = sequenceService.setSequence(sequenceData);
        this.setData({
          ...initialState, 
          loading: false
        });
        wx.hideLoading();
        wx.setNavigationBarTitle({ 
          title: `${getText(initialState.currentSequence.name)} - ${initialState.currentPoseIndex + 1}/${initialState.currentSequence.poses.length}` 
        });
      } else {
        console.error('[LOAD] Invalid sequence data:', sequenceData);
        throw new Error('加载的序列数据无效');
      }
    } catch (err) {
      console.error('[LOAD] Failed to load sequence:', err);
      let userErrorMessage = '无法加载序列数据，请稍后重试。';
      let toastMessage = '加载失败，请稍后重试';

      if (err && err.message === 'MISSING_SIGNED_URL') {
        userErrorMessage = '序列配置获取失败，请检查网络或稍后重试。';
        toastMessage = '序列配置获取失败';
      }
      
      this.setData({ 
        loading: false, 
        error: userErrorMessage, 
        currentSequence: null 
      });
      wx.hideLoading();
      wx.showToast({ title: toastMessage, icon: 'none' });
      wx.setNavigationBarTitle({ title: '加载错误' });
    }
  },

  // Timer management
  startTimer: function () {
    if (this.data.timerId) clearInterval(this.data.timerId);

    const timerId = setInterval(() => {
      if (this.data.timeRemaining > 0) {
        this.setData({ timeRemaining: this.data.timeRemaining - 1 });
      } else {
        clearInterval(this.data.timerId);
        this.setData({ timerId: null });
        if (this.data.isPlaying) {
          this.handleNext();
        }
      }
    }, 1000);
    this.setData({ timerId: timerId });
  },

  stopTimer: function () {
    if (this.data.timerId) {
      clearInterval(this.data.timerId);
      this.setData({ timerId: null });
    }
  },

  // Play audio guidance
  playAudioGuidance: function (src) {
    return new Promise((resolve, reject) => {
      if (!src) {
        console.warn('[AUDIO] No audio src provided');
        reject(new Error("No audio src provided"));
        return;
      }

      const audioCtx = wx.createInnerAudioContext({ useWebAudioImplement: false });
      audioCtx.src = src;
      audioCtx.onEnded(() => { 
        audioCtx.destroy(); 
        resolve(); 
      });
      audioCtx.onError((error) => {
        console.error('[AUDIO] Error playing:', src, error);
        wx.showToast({ title: '音频播放失败', icon: 'none' });
        audioCtx.destroy();
        reject(error);
      });
      audioCtx.play();
    });
  },

  // Navigation handlers
  handleBack: function () {
    this.stopTimer();
    wx.navigateBack();
  },

  handleNext: function () {
    this.stopTimer();
    const { currentSequence, currentPoseIndex } = this.data;
    const nextState = sequenceService.nextPose(currentSequence, currentPoseIndex);

    if (nextState) {
      this.setData({
        currentPoseIndex: nextState.currentPoseIndex_new,
        timeRemaining: nextState.timeRemaining_new
      });
      wx.setNavigationBarTitle({ 
        title: `${getText(currentSequence.name)} - ${nextState.currentPoseIndex_new + 1}/${currentSequence.poses.length}` 
      });
      
      if (this.data.isPlaying) {
        const newCurrentPose = currentSequence.poses[nextState.currentPoseIndex_new];
        this.playAudioGuidance(newCurrentPose.audioGuide)
          .catch(e => console.error("[AUDIO] Error in handleNext:", e));
        this.startTimer();
      }
    } else {
      wx.showToast({ title: '序列完成!', icon: 'success' });
      setTimeout(() => wx.redirectTo({ url: '/pages/index/index' }), 1500);
    }
  },

  // Toggle play/pause
  togglePlayPause: function () {
    const { isPlaying_new } = sequenceService.togglePlayPause(this.data.isPlaying);
    this.setData({ isPlaying: isPlaying_new });

    if (isPlaying_new) {
      const currentPose = this.data.currentSequence.poses[this.data.currentPoseIndex];
      this.playAudioGuidance(currentPose.audioGuide)
        .catch(e => console.error("[AUDIO] Error in togglePlayPause:", e));
      this.startTimer();
    } else {
      this.stopTimer();
    }
  },

  // Video selection handler
  handleChooseOrRecordVideo: function() {
    wx.chooseVideo({
      sourceType: ['album', 'camera'],
      compressed: false,
      maxDuration: 15,
      camera: 'back',
      success: (res) => {
        console.log("[VIDEO] Selected/recorded:", res);
        this.handleVideoValidation(res);
      },
      fail: (err) => {
        console.error("[VIDEO] wx.chooseVideo failed:", err);
        if (err.errMsg && err.errMsg.includes('cancel')) {
          wx.showToast({ title: '操作取消', icon: 'none' });
        } else {
          wx.showToast({ title: '选取视频失败', icon: 'none' });
        }
      }
    });
  },

  // Video validation
  handleVideoValidation: function(videoDetails) {
    console.log("[VALIDATION] Checking video:", videoDetails);

    if (videoDetails.duration > 15.5) {
      wx.showModal({ 
        title: '视频过长', 
        content: '您选择的视频超过15秒，请重新选取或录制一个较短的视频。', 
        showCancel: false, 
        confirmText: '知道了'
      });
      return;
    }

    const MAX_SIZE_BYTES = 10 * 1024 * 1024; // 10MB
    if (videoDetails.size > MAX_SIZE_BYTES) {
      wx.showModal({ 
        title: '视频文件过大', 
        content: '您选择的视频超过10MB，请重新选取或录制一个较小的视频。', 
        showCancel: false, 
        confirmText: '知道了'
      });
      return;
    }

    console.log("[VALIDATION] Video passed validation");
    this.setData({
      recordedVideo: videoDetails.tempFilePath,
      topThreeFrames: [],
      frameAnalysisResults: [],
      failedUploads: []
    });
    this.uploadAndScore();
  },

  // Image error handler - use placeholder
  onImageError: function(e) {
    const dataset = e.currentTarget.dataset;
    const imageType = dataset.type;
    const imageIndex = dataset.index;
    
    console.warn('[IMAGE_ERROR] Failed to load image:', e.detail.errMsg, 'Type:', imageType, 'Index:', imageIndex);
    
    // Update the specific image that failed based on type
    if (imageType === 'pose') {
      // Update current pose image
      if (this.data.currentSequence && this.data.currentSequence.poses[this.data.currentPoseIndex]) {
        this.setData({
          [`currentSequence.poses[${this.data.currentPoseIndex}].image_url`]: DEFAULT_POSE_IMAGE
        });
      }
    } else if (imageType === 'skeleton' && imageIndex !== undefined) {
      // Update skeleton image in topThreeFrames
      const frameIndex = parseInt(imageIndex);
      if (!isNaN(frameIndex) && this.data.topThreeFrames[frameIndex]) {
        this.setData({
          [`topThreeFrames[${frameIndex}].skeletonUrl`]: DEFAULT_POSE_IMAGE
        });
      }
    }
  },

  // Lifecycle hooks
  onHide: function () {
    this.stopTimer();
  },

  onUnload: function () {
    this.stopTimer();
    // Cancel any ongoing uploads
    if (this.data.currentUploadTasks && this.data.currentUploadTasks.length > 0) {
      console.log('[UNLOAD] Cancelling', this.data.currentUploadTasks.length, 'ongoing uploads');
      this.data.currentUploadTasks.forEach(task => {
        if (task && typeof task.abort === 'function') {
          task.abort();
        }
      });
    }
  }
});
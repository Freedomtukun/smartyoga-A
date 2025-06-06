// pages/photo-detect/photo-detect.js
import { DETECT_POSE_URL } from '../../utils/yoga-api.js';

Page({
  data: {
    imageUrl: '', // 用户上传的图片
    isDetecting: false, // 检测中状态
    hasError: false, // 错误状态
    errorMessage: '', // 错误信息
    showResult: false, // 显示结果
    // 检测结果数据
    skeletonUrl: '', // 骨架图
    score: 0, // 得分
    suggestions: '', // AI建议
    poseId: 'yoga_pose_001', // 默认姿势ID
    isSaving: false // 保存中状态
  },

  onLoad(options) {
    // 如果传入了姿势ID，使用传入的
    if (options.poseId) {
      this.setData({
        poseId: options.poseId
      });
    }
  },

  onShow() {
    wx.setNavigationBarTitle({
      title: '姿势检测'
    });
  },

  // 校验图片 URL 是否在白名单内
  isValidImageUrl(url) {
    if (!url) return false;
    const whitelist = /^https:\/\/static\.yogasmart\.cn\//;
    return whitelist.test(url) || /^wxfile:\/\//.test(url) || /^cloud:\/\//.test(url);
  },

  // 选择图片
  chooseImage() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['album', 'camera'],
      success: (res) => {
        const tempFilePath = res.tempFiles[0].tempFilePath;
        this.setData({
          imageUrl: tempFilePath,
          showResult: false,
          hasError: false
        });
      },
      fail: (err) => {
        console.error('选择图片失败', err);
        wx.showToast({ title: '选择图片失败', icon: 'none' });
      }
    });
  },

  // 开始检测图片
  startDetection() {
    // 未选择图片时给出提示
    if (!this.data.imageUrl) {
      wx.showToast({ title: '请先选择图片', icon: 'none' });
      return;
    }

    // 开启检测状态，按钮进入 loading
    this.setData({
      isDetecting: true,
      hasError: false
    });

    // 显示加载提示
    wx.showLoading({
      title: '检测中...',
      mask: true
    });

    // 上传图片并检测
    wx.uploadFile({
      url: DETECT_POSE_URL,
      filePath: this.data.imageUrl,
      name: 'file',
      formData: {
        poseId: this.data.poseId
      },
      header: {
        'Authorization': wx.getStorageSync('token') || ''
      },
      success: (res) => {
        wx.hideLoading();

        if (res.statusCode === 200) {
          try {
            const result = JSON.parse(res.data);

            if (result.code === 'OK') {
              // 验证返回的图片 URL 是否安全
              if (!this.isValidImageUrl(result.skeletonUrl)) {
                this.showError('返回的图片地址不安全');
                return;
              }

              // 检测成功，展示结果
              this.setData({
                isDetecting: false,
                showResult: true,
                skeletonUrl: result.skeletonUrl,
                score: result.score || 0,
                suggestions: (result.suggestions || '').trim()
              });

              // 保存到缓存，方便结果页读取
              this.saveToCache();

              // 滚动到结果区域
              wx.pageScrollTo({ selector: '#resultSection', duration: 300 });
            } else {
              // 业务错误
              this.showError(result.msg || '检测失败，请重试');
            }
          } catch (e) {
            console.error('解析响应失败', e);
            this.showError('服务器响应异常');
          }
        } else {
          // HTTP 错误
          this.showError(`请求失败(${res.statusCode})`);
        }
      },
      fail: (err) => {
        wx.hideLoading();
        console.error('上传失败', err);
        this.showError('上传失败，请检查网络');
      },
      complete: () => {
        this.setData({ isDetecting: false });
      }
    });
  },

  // 保存检测结果到缓存
  saveToCache() {
    const result = {
      imageUrl: this.data.skeletonUrl,
      score: this.data.score,
      suggestions: this.data.suggestions,
      timestamp: Date.now()
    };
    
    wx.setStorageSync('lastYogaResult', result);
  },

  // 显示错误信息
  showError(message) {
    this.setData({
      hasError: true,
      errorMessage: message,
      isDetecting: false
    });

    // 3秒后自动返回
    this.errorTimer = setTimeout(() => {
      wx.navigateBack();
    }, 3000);
  },

  // 立即返回
  goBackImmediately() {
    if (this.errorTimer) {
      clearTimeout(this.errorTimer);
    }
    wx.navigateBack();
  },

  // 预览原图
  previewOriginalImage() {
    if (this.data.imageUrl) {
      wx.previewImage({
        current: this.data.imageUrl,
        urls: [this.data.imageUrl]
      });
    }
  },

  // 预览骨架图
  previewSkeletonImage() {
    if (this.data.skeletonUrl) {
      wx.previewImage({
        current: this.data.skeletonUrl,
        urls: [this.data.skeletonUrl]
      });
    }
  },

  // 保存骨架图
  saveSkeletonImage() {
    if (!this.data.skeletonUrl) {
      wx.showToast({
        title: '图片不存在',
        icon: 'none'
      });
      return;
    }

    this.setData({ isSaving: true });
    wx.showLoading({ title: '保存中...' });

    // 获取保存权限
    wx.getSetting({
      success: (res) => {
        if (!res.authSetting['scope.writePhotosAlbum']) {
          wx.authorize({
            scope: 'scope.writePhotosAlbum',
            success: () => {
              this.doSaveImage();
            },
            fail: () => {
              wx.hideLoading();
              this.setData({ isSaving: false });
              wx.showModal({
                title: '提示',
                content: '需要您的相册权限才能保存图片',
                showCancel: true,
                confirmText: '去设置',
                success: (modalRes) => {
                  if (modalRes.confirm) {
                    wx.openSetting();
                  }
                }
              });
            }
          });
        } else {
          this.doSaveImage();
        }
      }
    });
  },

  // 执行保存图片
  doSaveImage() {
    const url = this.data.skeletonUrl;
    const save = (filePath) => {
      wx.saveImageToPhotosAlbum({
        filePath,
        success: () => {
          this.setData({ isSaving: false });
          wx.hideLoading();
          wx.showToast({ title: '保存成功', icon: 'success' });
        },
        fail: this.handleSaveError
      });
    };

    if (url.startsWith('http')) {
      wx.downloadFile({
        url,
        success: (res) => {
          if (res.statusCode === 200) {
            save(res.tempFilePath);
          } else {
            this.handleSaveError();
          }
        },
        fail: this.handleSaveError
      });
    } else {
      save(url);
    }
  },

  // 保存失败处理
  handleSaveError() {
    this.setData({ isSaving: false });
    wx.hideLoading();
    wx.showModal({
      title: '保存失败',
      content: '您也可以长按图片直接保存',
      showCancel: false
    });
  },

  // 重新检测
  resetDetection() {
    this.setData({
      imageUrl: '',
      showResult: false,
      skeletonUrl: '',
      score: 0,
      suggestions: '',
      hasError: false
    });
  },

  // 查看完整结果
  viewFullResult() {
    wx.redirectTo({
      url: `/pages/result/result?imageUrl=${encodeURIComponent(this.data.skeletonUrl)}&score=${this.data.score}&suggestions=${encodeURIComponent(this.data.suggestions)}`
    });
  },

  // 分享配置
  onShareAppMessage() {
    if (this.data.showResult) {
      return {
        title: `我的姿势得分${this.data.score}分，快来试试吧！`,
        path: '/pages/index/index',
        imageUrl: this.data.skeletonUrl
      };
    }
    return {
      title: 'AI姿势检测，快来试试吧！',
      path: '/pages/index/index'
    };
  },

  // 页面卸载清理
  onUnload() {
    if (this.errorTimer) {
      clearTimeout(this.errorTimer);
    }
  }
});
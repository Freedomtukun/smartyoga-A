// pages/result/result.js
Page({
  data: {
    imageUrl: '',
    score: 0,
    suggestions: '',
    isLoading: false,
    hasError: false,
    errorMessage: '',
    // 分享卡片相关预留
    shareCardUrl: '',
    isGeneratingCard: false
  },

  onLoad(options) {
    // 修复：使用 hasOwnProperty 判断 score 参数
    if (options.imageUrl && options.hasOwnProperty('score')) {
      // 增加图片URL安全性检查
      const imageUrl = decodeURIComponent(options.imageUrl);
      if (!this.isValidImageUrl(imageUrl)) {
        this.showError('图片地址无效');
        return;
      }
      
      this.setData({
        imageUrl: imageUrl,
        score: parseInt(options.score) || 0,
        suggestions: decodeURIComponent(options.suggestions || '').trim() // 添加 trim
      });
      
      // 保存到历史记录
      this.saveToHistory();
    } else {
      // 尝试从缓存获取
      this.loadFromCache();
    }
  },

  onShow() {
    // 设置导航栏标题
    wx.setNavigationBarTitle({
      title: '检测结果'
    });
  },

  // 验证图片URL是否合法
  isValidImageUrl(url) {
    if (!url) return false;
    // 支持 http/https 和微信临时文件
    return /^https?:\/\//.test(url) || /^wxfile:\/\//.test(url) || /^cloud:\/\//.test(url);
  },

  // 从缓存加载数据
  loadFromCache() {
    try {
      const cachedResult = wx.getStorageSync('lastYogaResult');
      if (cachedResult && cachedResult.imageUrl) {
        this.setData({
          imageUrl: cachedResult.imageUrl,
          score: cachedResult.score || 0,
          suggestions: (cachedResult.suggestions || '').trim()
        });
      } else {
        this.showError('未找到检测结果');
      }
    } catch (e) {
      this.showError('加载结果失败');
    }
  },

  // 显示错误信息
  showError(message) {
    this.setData({
      hasError: true,
      errorMessage: message
    });

    // 3秒后自动返回
    this.errorTimer = setTimeout(() => {
      wx.navigateBack();
    }, 3000);
  },

  // 立即返回（新增）
  goBackImmediately() {
    if (this.errorTimer) {
      clearTimeout(this.errorTimer);
    }
    wx.navigateBack();
  },

  // 保存到历史记录（新增）
  saveToHistory() {
    try {
      const history = wx.getStorageSync('yogaHistory') || [];
      const newRecord = {
        imageUrl: this.data.imageUrl,
        score: this.data.score,
        suggestions: this.data.suggestions,
        timestamp: Date.now()
      };
      
      // 避免重复
      const exists = history.some(item => 
        item.imageUrl === newRecord.imageUrl && 
        item.score === newRecord.score
      );
      
      if (!exists) {
        history.unshift(newRecord);
        // 最多保存20条
        if (history.length > 20) {
          history.splice(20);
        }
        wx.setStorageSync('yogaHistory', history);
      }
    } catch (e) {
      console.error('保存历史记录失败', e);
    }
  },

  // 保存图片到相册
  saveImage() {
    if (!this.data.imageUrl) {
      wx.showToast({
        title: '图片不存在',
        icon: 'none'
      });
      return;
    }

    this.setData({ isLoading: true });

    // 先获取授权
    wx.getSetting({
      success: (res) => {
        if (!res.authSetting['scope.writePhotosAlbum']) {
          wx.authorize({
            scope: 'scope.writePhotosAlbum',
            success: () => {
              this.doSaveImage();
            },
            fail: () => {
              this.setData({ isLoading: false });
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
    // 如果是网络图片，需要先下载
    if (this.data.imageUrl.startsWith('http')) {
      wx.downloadFile({
        url: this.data.imageUrl,
        success: (res) => {
          if (res.statusCode === 200) {
            this.saveLocalImage(res.tempFilePath);
          } else {
            this.handleSaveError();
          }
        },
        fail: () => {
          this.handleSaveError();
        }
      });
    } else {
      // 本地图片直接保存
      this.saveLocalImage(this.data.imageUrl);
    }
  },

  // 保存本地图片
  saveLocalImage(filePath) {
    wx.saveImageToPhotosAlbum({
      filePath: filePath,
      success: () => {
        this.setData({ isLoading: false });
        wx.showToast({
          title: '保存成功',
          icon: 'success',
          duration: 2000
        });
      },
      fail: (err) => {
        this.handleSaveError();
      }
    });
  },

  // 处理保存失败（优化提示）
  handleSaveError() {
    this.setData({ isLoading: false });
    wx.showModal({
      title: '保存失败',
      content: '您也可以长按图片直接保存哦~',
      showCancel: false,
      confirmText: '知道了'
    });
  },

  // 分享功能（标注开发中）
  onShare() {
    wx.showToast({
      title: '功能开发中',
      icon: 'none',
      duration: 1500
    });
  },

  // 跳转到邀请码/推广入口页面
  onInviteClick() {
    wx.navigateTo({
      url: '/pages/invite/invite'
    });
  },

  // 图片预览
  previewImage() {
    if (this.data.imageUrl) {
      wx.previewImage({
        current: this.data.imageUrl,
        urls: [this.data.imageUrl]
      });
    }
  },

  // 返回上一页
  goBack() {
    wx.navigateBack();
  },

  // 重新检测（使用 reLaunch 避免页面栈过深）
  retryDetection() {
    wx.reLaunch({
      url: '/pages/upload/upload'
    });
  },

  // 分享给朋友
  onShareAppMessage() {
    const score = this.data.score;
    let title = `我的瑜伽姿势得分${score}分`;
    
    // 根据分数给出不同的分享文案
    if (score >= 90) {
      title += '，已经是瑜伽大师了！';
    } else if (score >= 80) {
      title += '，表现很棒！';
    } else if (score >= 70) {
      title += '，继续加油！';
    } else {
      title += '，一起来练习吧！';
    }
    
    return {
      title: title,
      path: '/pages/index/index',
      imageUrl: this.data.imageUrl
    };
  },

  // 分享到朋友圈
  onShareTimeline() {
    return {
      title: `瑜伽姿势检测得分${this.data.score}分`,
      query: '',
      imageUrl: this.data.imageUrl
    };
  },

  // 页面卸载时清理
  onUnload() {
    if (this.errorTimer) {
      clearTimeout(this.errorTimer);
    }
  }
});
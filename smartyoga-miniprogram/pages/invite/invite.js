Page({
    data: {
      inviteCode: '',     // 用户输入的邀请码
      myInviteCode: '',   // 本人专属邀请码
      // qrcodeUrl: ''    // 预留：生成二维码图片URL
    },
  
    // 输入框实时更新
    onInput(e) {
      this.setData({ inviteCode: e.detail.value.trim() })
    },
  
    // 绑定邀请码（注册/归属上级）
    onBind() {
      if (!this.data.inviteCode) {
        wx.showToast({ title: '请输入邀请码', icon: 'none' })
        return
      }
      wx.cloud.callFunction({
        name: 'registerUser',
        data: {
          name: '用户昵称', // 可用实际昵称替换
          inviteCode: this.data.inviteCode
        },
        success: res => {
          if (res.result.success) {
            wx.showToast({ title: '绑定成功', icon: 'success' })
            // 可跳转/返回，也可以刷新页面状态
          } else {
            wx.showToast({ title: res.result.msg, icon: 'none' })
          }
        }
      })
    },
  
    // 生成并展示专属邀请码
    onGenCode() {
      // 获取当前用户ID，实际项目应替换为登录后获取到的ID
      const userId = wx.getStorageSync('userId') || '测试用户ID'
      wx.cloud.callFunction({
        name: 'genInviteCode',
        data: { userId },
        success: res => {
          if (res.result.success) {
            this.setData({ myInviteCode: res.result.inviteCode })
            // 预留二维码生成逻辑
          } else {
            wx.showToast({ title: res.result.msg, icon: 'none' })
          }
        }
      })
    }
  })
  
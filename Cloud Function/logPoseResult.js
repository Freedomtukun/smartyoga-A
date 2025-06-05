// 腾讯云函数 - COS触发器
// 当有新的骨架图上传到COS时，记录到数据库

const TcbRouter = require('tcb-router');
const cloud = require('wx-server-sdk');

// 初始化云开发
cloud.init({
    env: cloud.DYNAMIC_CURRENT_ENV
});

const db = cloud.database();
const _ = db.command;

/**
 * 解析COS对象key，提取poseId和时间戳
 * @param {string} key - 如 'skeletons/mountain_pose_1234567890.png'
 * @returns {object} {poseId, timestamp}
 */
function parseKey(key) {
    try {
        // 移除前缀和后缀
        const filename = key.replace('skeletons/', '').replace('.png', '');
        
        // 找到最后一个下划线的位置
        const lastUnderscore = filename.lastIndexOf('_');
        if (lastUnderscore === -1) {
            throw new Error('Invalid key format');
        }
        
        const poseId = filename.substring(0, lastUnderscore);
        const timestamp = parseInt(filename.substring(lastUnderscore + 1));
        
        return { poseId, timestamp };
    } catch (error) {
        console.error('解析key失败:', key, error);
        return null;
    }
}

/**
 * 主函数 - COS触发器入口
 */
exports.main = async (event, context) => {
    console.log('COS触发事件:', JSON.stringify(event));
    
    try {
        // 从COS事件中获取信息
        const { Records } = event;
        if (!Records || Records.length === 0) {
            console.log('没有记录需要处理');
            return { code: 0, message: 'No records to process' };
        }
        
        const results = [];
        
        for (const record of Records) {
            try {
                // 获取COS对象信息
                const bucket = record.cos.cosBucket.name;
                const key = record.cos.cosObject.key;
                const size = record.cos.cosObject.size;
                const eventName = record.event.eventName;
                
                console.log(`处理对象: ${key}, 事件: ${eventName}`);
                
                // 只处理PUT事件（新上传）
                if (!eventName.includes('Put')) {
                    continue;
                }
                
                // 解析key获取poseId和时间戳
                const parsed = parseKey(key);
                if (!parsed) {
                    console.error('无法解析key:', key);
                    continue;
                }
                
                const { poseId, timestamp } = parsed;
                
                // 构建骨架图URL
                const skeletonUrl = `https://${bucket}.cos.${record.cos.cosBucket.region}.myqcloud.com/${key}`;
                
                // 从event中获取用户信息（需要在上传时通过自定义header传递）
                // 注意：实际使用时需要确保有用户认证机制
                const openId = event.openId || 'unknown_user';
                const score = event.score || 0;
                
                // 写入数据库
                const data = {
                    openId: openId,
                    poseId: poseId,
                    score: score,
                    skeletonUrl: skeletonUrl,
                    timestamp: timestamp,
                    createTime: db.serverDate(),
                    // 额外信息
                    bucketName: bucket,
                    objectKey: key,
                    objectSize: size
                };
                
                const res = await db.collection('pose_history').add({
                    data: data
                });
                
                console.log('数据写入成功:', res._id);
                results.push({
                    success: true,
                    id: res._id,
                    key: key
                });
                
                // 可选：发送推送通知给用户
                if (openId !== 'unknown_user' && score > 0) {
                    await sendNotification(openId, poseId, score);
                }
                
            } catch (error) {
                console.error('处理记录失败:', error);
                results.push({
                    success: false,
                    error: error.message,
                    key: record.cos.cosObject.key
                });
            }
        }
        
        return {
            code: 0,
            message: 'Processing completed',
            results: results
        };
        
    } catch (error) {
        console.error('云函数执行失败:', error);
        return {
            code: -1,
            message: error.message
        };
    }
};

/**
 * 发送评分通知给用户（可选功能）
 */
async function sendNotification(openId, poseId, score) {
    try {
        // 获取用户的订阅消息配置
        const userInfo = await db.collection('users')
            .where({ openId: openId })
            .limit(1)
            .get();
        
        if (userInfo.data.length === 0) {
            console.log('用户未找到:', openId);
            return;
        }
        
        const user = userInfo.data[0];
        
        // 检查用户是否订阅了通知
        if (!user.subscribeNotification) {
            return;
        }
        
        // 构建消息内容
        const templateId = 'YOUR_TEMPLATE_ID'; // 替换为实际的模板ID
        const page = `pages/result/result?poseId=${poseId}&score=${score}`;
        const data = {
            thing1: { value: getPoseName(poseId) }, // 动作名称
            number2: { value: score.toString() },   // 得分
            thing3: { value: getScoreComment(score) }, // 评价
            time4: { value: formatTime(new Date()) }   // 时间
        };
        
        // 发送订阅消息
        const result = await cloud.openapi.subscribeMessage.send({
            touser: openId,
            templateId: templateId,
            page: page,
            data: data,
            miniprogramState: 'formal'
        });
        
        console.log('通知发送成功:', result);
        
    } catch (error) {
        console.error('发送通知失败:', error);
    }
}

/**
 * 获取动作的中文名称
 */
function getPoseName(poseId) {
    const names = {
        'mountain_pose': '山式',
        'warrior_pose': '战士式',
        'tree_pose': '树式',
        'downward_dog': '下犬式',
        'plank_pose': '平板支撑'
    };
    return names[poseId] || poseId;
}

/**
 * 根据分数生成评价
 */
function getScoreComment(score) {
    if (score >= 90) return '完美！';
    if (score >= 80) return '很棒！';
    if (score >= 70) return '不错！';
    if (score >= 60) return '继续加油！';
    return '需要改进';
}

/**
 * 格式化时间
 */
function formatTime(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hour = String(date.getHours()).padStart(2, '0');
    const minute = String(date.getMinutes()).padStart(2, '0');
    return `${year}-${month}-${day} ${hour}:${minute}`;
}
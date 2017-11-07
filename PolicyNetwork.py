# region 加载库，先测试一把，并配置参数
import numpy as np
import tensorflow as tf
import gym
# 创建环境env
env = gym.make('CartPole-v0')

# 测试随机Action的表现
# 初始化环境
env.reset()
random_episodes = 0
reward_sum = 0
while random_episodes < 10: #　玩10把
    # 渲染图像
    env.render()
    # 使用np.random.randint(0,2)产生随机的Action
    # env.step()执行这个而Action.
    observation, reward, done, _ = env.step(np.random.randint(0,2))
    # 累加到这把的总奖励里
    reward_sum += reward
    # done为True,即任务失败,则实验结束
    if done:
        # 展示这次试验累计的奖励
        random_episodes += 1
        print("Reward for this episode was:",reward_sum)
        reward_sum = 0
        # 初始化环境
        env.reset()
        
# hyperparameters
H = 50                     # 50个隐层神经元
batch_size = 25            # every how many episodes to do a param update?
learning_rate = 1e-1       # 学习速率
gamma = 0.99               # reward的discount比例，要＜１，防止reward被无损耗地不断累加导致发散．未来奖励不确定性必须打折
D = 4                      # input dimensionality

# endregion

tf.reset_default_graph()

# region 定义计算图结构
#This defines the network as it goes from taking an observation of the environment to 
#giving a probability of chosing to the action of moving left or right.

# 环境信息observations作为输入信息，最后输出一个概率值，用以选择Action
# 我们只有２个action,向左加力，向右加力，因此可以通过一个概率值决定．
observations = tf.placeholder(tf.float32, [None,D] , name="input_x")
W1 = tf.get_variable("W1", shape=[D, H],
           initializer=tf.contrib.layers.xavier_initializer())
# 隐层layer1，注意没有偏置．
layer1 = tf.nn.relu(tf.matmul(observations,W1))
W2 = tf.get_variable("W2", shape=[H, 1],
           initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1,W2)
# sigmoid层．即输出层.
probability = tf.nn.sigmoid(score)

# 定义loss
input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
advantages = tf.placeholder(tf.float32,name="reward_signal")
# action=1时,我们定义的高仿标签y=0,则loglik=tf.log(probability)
# action=0时,我们定义的高仿标签y=1,这时loglik=tf.log(1-probability)
# 而action=1本来就是由概率probability产生的,见执行部分.
# 所以loglik其实就时当前action对应的概率的对数.
# 因为我们不确定此时action是1还是0,所以用了y*(y-p)+(1-y)*(y+p)这么繁琐的表达.
loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))
# advantages是每次成功完成任务后,对之前每步action的价值评估
# 为了减小loss,就要让获得较多advantages的action的概率变大.
loss = -tf.reduce_mean(loglik * advantages)
# 定义梯度
# 获取全部可训练参数tvars
tvars = tf.trainable_variables()
# 针对loss，计算tvars的梯度
# tvars是自变量,loss是因变量,
newGrads = tf.gradients(loss,tvars)

# 定义优化器
adam = tf.train.AdamOptimizer(learning_rate=learning_rate) # Our optimizer
# 定义更新操作
W1Grad = tf.placeholder(tf.float32,name="batch_grad1")
W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
batchGrad = [W1Grad,W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))


# endregion

# 用来估算每一个Action对应的潜在价值discount_r
def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        # r[0]就是gg时的动作价值,r[1]就是gg前的动作价值,最后的r[]才是本步的动作价值
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r



# xs是环境信息observation的列表
# ys是人为定义的label列表
# drs是每一个Action的Reward
xs,ys,drs = [],[],[]
reward_sum = 0              # 累计的reward
episode_number = 1
total_episodes = 10000      # 总的试验次数

# region 执行计算图
# Launch the graph
with tf.Session() as sess:
    # 一开始模型欠成熟,没必要观察,且render会带来比较大的延迟,所以关闭render
    rendering = False
    init = tf.global_variables_initializer()
    sess.run(init)
    observation = env.reset() # Obtain an initial observation of the environment

    # Reset the gradient placeholder. We will collect gradients in 
    # gradBuffer until we are ready to update our policy network. 
    # 执行tvars,用来创建储存参数梯度的缓冲器gradBuffer
    # 获取参数的目的是为了获取参数数目,因此之后要对gradBuffer进行清零.
    gradBuffer = sess.run(tvars)
    # 把gradBuffer全部初始化为0.
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    
    while episode_number <= total_episodes:# 当试验小于10000次时
        
        # Rendering the environment slows things down, 
        # so let's only look at it once our agent is doing a good job.
        # 当某个batch的平均奖励大于100,即agent表现良好,就调用render()进行展示
        # 一旦有一个表现良好,之后都会进行展示.
        if reward_sum/batch_size > 100 or rendering == True : 
            env.render()
            rendering = True
            
        # Make sure the observation is in a shape the network can handle.
        # 将observation变形为策略网络输入的格式x,传入网络中.
        x = np.reshape(observation,[1,D])
        
        # agent网络拿到环境送过来的x,产生一个probability,生成一定概率分布的action
        tfprob = sess.run(probability,feed_dict={observations: x})
        action = 1 if np.random.uniform() < tfprob else 0
        
        xs.append(x) # observation
        # 在这里,y=1-action,y就是fake label
        y = 1 if action == 0 else 0
        ys.append(y)

        # 执行一次action,环境返回环境状态和奖励
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

        # done=true,则进入试验结束时的处理
        if done: 
            episode_number += 1
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            # 将列表xs,ys,drs的元素纵向堆叠起来,并清空以备下次试验使用.
            # epx,epy,epr即为一次试验中获得的所有observation,label,reward列表
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            xs,ys,drs = [],[],[] # reset array memory

            # compute the discounted reward backwards through time
            # 计算每一步action的潜在价值.
            # 类似于革命成功了,就要好好复盘,之前每一个action到底价值几何.
            # 向dicount_rewards()里传入这本次reward的列表epr,就能获得一个复盘表(即一个包含每个action到底多重要的列表)
            discounted_epr = discount_rewards(epr)
            # size the rewards to be unit normal (helps control the gradient estimator variance)
            # 分布稳定的discounted_reward有利于训练的稳定
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            
            # Get the gradient for this episode, and save it in the gradBuffer
            # 使用操作newGrads求解梯度.
            tGrad = sess.run(newGrads,feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            # 累加梯度到gradBuffer中
            for ix,grad in enumerate(tGrad):
                gradBuffer[ix] += grad
                
            # If we have completed enough episodes, then update the policy network with our gradients.
            # 当试验次数达到batch_size的整数倍时,gradBuffer就累计了足够多的梯度.
            if episode_number % batch_size == 0:  # 每获得25次成功之时
                # 这时,使用updateGrads操作,将gradBuffer中的梯度更新到策略网络的模型参数中.
                sess.run(updateGrads,feed_dict={W1Grad: gradBuffer[0],W2Grad:gradBuffer[1]})
                # 清空gradBuffer,为计算下一个batch做准备
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
                
                # Give a summary of how well our network is doing for each batch of episodes.
                # running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                # 展示当前的试验次数episode_number,batch内每次试验平均获得的reward
                print('Average reward for episode %d : %f.' % (episode_number,reward_sum/batch_size))

                # 如果batch内每次试验平均获得的reward大于200,就成功完成了任务,终止循环.
                if reward_sum/batch_size > 200: 
                    print("Task solved in",episode_number,'episodes!')
                    break
                # 如果没有达到目标,则清空reward_sum,重新累计下一个batch的总reward
                reward_sum = 0

            # 每次试验结束后,将任务环境env重置,方便下一次试验
            observation = env.reset()
        
# endregion

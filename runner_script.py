import os


def make_command(env, alg, grad, num_seeds=3, seed_offset=0): 
	base_tmux = "tmux new-session -d -s "
	session_name = env+'_'+alg+ '_grad_' + str(grad)
	seed_list = [str(i + seed_offset) for i in range(num_seeds)]
	seed_str = ' --seed ' + str.join(' ', seed_list)
	run_command = 'python -m spinup.run ' + alg + " --env " + env + " --use_grad_penalty " + str(grad) + " --exp_name " + session_name + seed_str
	full_command = base_tmux + session_name + " \'" + run_command + "\'"
	return full_command

envs = ["AntPyBulletEnv-v0", "HalfCheetahPyBulletEnv-v0", "HopperPyBulletEnv-v0", "HumanoidPyBulletEnv-v0", "Walker2DPyBulletEnv-v0"]
algs = ['sac', 'td3']
grad_values = [True, False]
alg = algs[0]
env = envs[2]
for val in grad_values:
	os.system(make_command(env, alg, val))
	# print(make_command(env, alg, val))
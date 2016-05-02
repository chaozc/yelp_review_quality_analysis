import json
import subprocess

def delete_keys_for_dic(dic, keys):
	for key in keys:
		dic.pop(key, None)
	return dic

def delete_keys_for_json_file(infname, oufname, keys):
	inf = open(infname, 'r')
	ouf = open(oufname, 'w')
	for line in inf:
		js = json.loads(line)
		js = delete_keys_for_dic(js, keys)
		jsStr = json.dumps(js)+'\n'
		ouf.write(jsStr)
	inf.close()
	ouf.close()


def join(infname1, infname2, oufname, join_key, new_key):
	inf2 = open(infname2, 'r')
	dic = {}
	for line in inf2:
		js = json.loads(line)
		dic[js[join_key]] = js
	inf2.close()
	print("%s finish", infname2)	

	inf1 = open(infname1, 'r')
	ouf = open(oufname, 'w')
	for line in inf1:
		js = json.loads(line)
		v = js[join_key]
		js = delete_keys_for_dic(js, [join_key])
		js[new_key] = dic[v]
		jsStr = json.dumps(js)+'\n'
		ouf.write(jsStr)
	inf1.close()
	ouf.close()

if __name__ == "__main__":
	#subprocess.call(["grep", "2015-", "../data/yelp_academic_dataset_review.json", ">", "../data/review_2015.json"])
	delete_keys_for_json_file("../data/yelp_academic_dataset_user.json", "../data/user_clean.json", ["friends"])
	join("../data/review_2013.json", "../data/user_clean.json", "../data/review_user_2013.json", "user_id", "user")
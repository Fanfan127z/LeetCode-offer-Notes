## **LeetCode 《剑指offer（第2版）》 刷题笔记**

这里的题目都是我自己一刷时没do出来的（看题解） or do的有些磕磕绊绊需要想挺久才do出来的题目！！！（暗示我自己这些剑指offer的题目为肯定是需要2刷甚至3or4刷的！）

我到时候面试的时候就一定要先把==10大排序算法==再背回来！清除知道了解他们的时间空间复杂度！==(這一點是我在刷lc题目的时候没注意到的！)==

### 目前累计总共有==《29》==道题：

#### <1> [LeetCode 剑指offer 04.二维数组中的查找：](https://leetcode.cn/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

题解：

```c++
// 学习题解区的高赞题解！
// 来自B站Up主:香辣鸡排蛋包饭 小姐姐的视频！
class Solution {
public:
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        int i = matrix.size()-1,j = 0;
        while(i >= 0 && j < matrix[0].size()){
            if(matrix[i][j] < target)j++;
            else if(matrix[i][j] > target)i--;
            else return true;
        }
        return false;
    }
};
```

#### <2> [LeetCode 剑指offer 25.合并两个排序的链表：](https://leetcode.cn/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/)

题解：

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */

class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* newList = new ListNode(-1);
        ListNode* cur = newList;
        // 这道题目就是整体上思路需要理清楚才能够一次性写出来无wrong的codes!
            /*  整体思路：
                case1:l1空 且 l2空 == 直接break跳出构建新list的while循环
                case2:l1不空 且 l2空 == 先把l1剩余的all元素拿来构建新list然后再break
                case3:l1空 且 l2不空 == 先把l2剩余的all元素拿来构建新list然后再break
                case4:l1不空 且 l2不空
                    case5: l1->val < l2->val == 先把l1当前的元素拿来构建新list然后再continue
                    case6: l1->val >= l2->val == 先把l2当前的元素拿来构建新list然后再continue
            */ 
        while(true){
            if(!l1 && !l2) break;
            else if(l1 && !l2){
                while(l1){
                    ListNode* newNode = new ListNode(l1->val);
                    cur->next = newNode;
                    cur = cur->next;
                    l1 = l1->next;
                }
                break;
            }else if(!l1 && l2){
                while(l2){
                    ListNode* newNode = new ListNode(l2->val);
                    cur->next = newNode;
                    cur = cur->next;
                    l2 = l2->next;
                }
                break;
            }
            else if(l1 && l2){
                ListNode* newNode = new ListNode();
                if(l1->val < l2->val){
                    newNode->val = l1->val;
                    cur->next = newNode;
                    l1 = l1->next;
                }else{
                    newNode->val = l2->val;
                    cur->next = newNode;
                    l2 = l2->next;
                }
            }
            cur = cur->next;
        }
        return newList->next;
    }
};
```

#### <3> [LeetCode 剑指offer 31.栈的压入、弹出序列：](https://leetcode.cn/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)

（这个题目我想到了lc高赞答案的题解了，但是没有自己写出来！）

```c++
// time:O(n) 一个for
// space:O(n) st最大可以n个元素
/*
	总体思路：借助一个stack来do，模拟压栈和出栈的case即可！
	压栈就按照pushed数组的顺序压入即可，然后同时将st.top()头部元素与popped元素依次顺序do比较
	if st.top()头部元素与popped的当前元素 相等 的话
		st.pop()并且让指向popped的下标后移
	else 继续遍历pushed数组并将其元素push进st中
	最终判断st是否为空即可判断是否为正确的压栈出栈操作序列了！
	if st.empty() == true ==> return true
	if st.empty() == false ==> return false
*/
class Solution {
public:
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        // 拉个辅助stack即可了！
        stack<int> st;
        int idx_poped = 0;
        for(int i = 0;i < pushed.size();++i){
            st.push(pushed[i]);
            while(!st.empty() && st.top() == popped[idx_poped]){
                idx_poped++;
                st.pop();
            }
        }
        return st.empty();
    }
};
```



#### <4> [LeetCode 剑指offer 50.第一个只出现一次的字符：](https://leetcode.cn/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/)

```c++
// 暴力法的话就太tmd简单了（这里就不写了，也就2层for循环就可以暴力出来了！）！
/*
    思路：
    	step1: 遍历字符串s，先用哈希表来记录s中哪些字符是只会出现一次的，是的就true，否的就false。
    	step2: 再遍历一次字符串s，看哈希表中记录着的哪个只出现一次的字符（即it->second == true的字符）处在s中的最前面，谁处在最前面就代表着是s字符串中的第一个只出现一次的字符了。
*/
// 因为hashMap只会存储不重复key的pair对，又因为s字符串只有小写字母，
// 因此只会存在26种可能，so空间复杂度为O(26) == O(1)
// 时间复杂度上，遍历s是O(2*n) == O(n),n是s字符串的长度,而map的操作的时间复杂度只是O(1)而已！
// time:O(n)
// space:O(1)
class Solution {
public:
    char firstUniqChar(string s) {
        unordered_map<char,bool> hashMap;
        for(char& ch : s){
            auto it = hashMap.find(ch);
            if(it != hashMap.end())it->second = false;
            else hashMap[ch] = true;
            // 一句代码也可以这么写：
            // hashMap[ch] = hashMap.find(ch) == hashMap.end();
        }
        for(char& ch : s){
            // 字符串s中谁先出现一次就返回谁！
            if(hashMap[ch])return ch;
        }
        return ' ';
    }
};
```



#### <5> [LeetCode 剑指offer 35.复杂链表的复制：](https://leetcode.cn/problems/fu-za-lian-biao-de-fu-zhi-lcof/)

这是lc剑指offer板块中的==评论区高赞大佬Krahets==教我的题解：

![lc剑指offer35](F:\学习过程中的截图\lc剑指offer35.png)

```c++
// time:O(n)
// space:O(n)
/* 整体思路：
            分2步走：
            1-new出新list的all节点(因为是复制copy，那肯定要new出来的！)并用hashMap记录new的all新节点
            2-利用前面hashMap记录new的all新节点，拿这些节点来do新的list连线工作（即上图中的虚线部分，）！
        */
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if(head == nullptr)return head;
        unordered_map<Node*,Node*> hashMap;
        // key:存放的是旧的list节点
        // val:存放的是新的list节点
        Node* cur = head;
        while(cur){// 这一步只是new出来all的全新的list的节点而已！
            hashMap[cur] = new Node(cur->val);
            cur = cur->next;
        }
        cur = head;
        while(cur){// 这一步给这个新的list的各个节点连上线！
        // 包括连上next和random的线！
            hashMap[cur]->next = hashMap[cur->next];
            hashMap[cur]->random = hashMap[cur->random];
            cur = cur->next; 
        }
        return hashMap[head];
    }
};
```

#### <6> [LeetCode 剑指offer 54.二叉搜索树的第k大节点：](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
 /*解题思路：
 	已知：二叉搜索树的中序遍历是递增序列(这可是Data Structure中的常识)
 	那么求二叉搜索树的第k大节点的值 <==> 求其中序遍历的逆序（递减序列）的第k个节点的值！
  	你拿lc官方给出的二叉搜索树画一画举个例子就知道了哈！
 */ 
class Solution {
public:
    vector<int> tmp;
    void reverse_inOrder(TreeNode* cur){
        if(cur == nullptr)return;
        // 右 中 左
        reverse_inOrder(cur->right);
        tmp.push_back(cur->val);
        reverse_inOrder(cur->left);
        return;
    }
    int kthLargest(TreeNode* root, int k) {
        tmp.clear();
        reverse_inOrder(root);
        return tmp[k-1];// 下标是从0开始，而第几个元素时从1开始计数的！so要减去1！
    }
};
// 版本2：（我完全理解了上面的代码之后，自己写出来的）
class Solution {
public:
    int res = 0,cnt = 0;
    void midT(TreeNode* cur,int k){
        // 递归终止条件
        if(cur==nullptr)return ;
        // 单层递归logic
        // 右 中 左 的logic
        midT(cur->right,k);// left
        // mid
        cnt++;
        if(cnt == k){
            res = cur->val;// 这就是BST的第k大节点的值！
            return;
        }
        midT(cur->left,k);// right
    }
    int kthLargest(TreeNode* root, int k) {
        // 倒中序遍历，找到的第k个节点其实就是BST的第k大节点的值了！
        if(root==nullptr)return -1;
        res=0,cnt=0;
        midT(root,k);
        return res;
    }
};
```



#### <7> [LeetCode 剑指offer 55II.平衡二叉树：](https://leetcode.cn/problems/ping-heng-er-cha-shu-lcof/)

这道题目为在carl哥的网站上刷了N次了，但是刷剑指offer时候还是有点陌生，==中途遍历到发现左右子树是非平衡就直接返回-1表示整棵树是非平衡这2行代码为没有写出来！！！==

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    // 判断一棵BT是否为一棵二叉平衡树
    // 求BT的深度 == BT的前序遍历的logic了！
    // 但是这个题目貌似用后续遍历的方式更加容易理解！
    // 用int值-1表示该BT是非平衡二叉树！
    int traversal(TreeNode* cur){
        if(cur == nullptr)return 0;
        int left = traversal(cur->left);
        if(left == -1)return -1;// 若中途遍历BT的时候发现有非平衡的case也要马上返回！
        int right = traversal(cur->right);
        if(right == -1)return -1;// 若中途遍历BT的时候发现有非平衡的case也要马上返回！
        if(abs(left - right) > 1)return -1;
        int mid = max(left,right)+1;
        return mid;// 返回当前BT的中节点为根的BT的深度！
    }
    bool isBalanced(TreeNode* root) {
        if(root == nullptr)return true;// 空树也属于平衡树的一种！
        return traversal(root) != -1;
    }
};
```



#### <8> [LeetCode 剑指offer 56I.数组中数字出现的次数](https://leetcode.cn/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)



```c++
// time:O(n)
// spcae:O(1)
class Solution {
public:
    vector<int> singleNumbers(vector<int>& nums) {
        // If用哈希表法，则是双O(n)的算法！因为hashMap需要存储nums.size()个元组对！
        // so我自己想了一种办法！
        // 先sort升序排序方便找出重复元素然后一个while循环就deal问题了！
        // 可以拿[1 1 4 6] or [1 1 2 3 3 4 4 10] 来画图举例子验证我写的codes！
        sort(nums.begin(),nums.end());// sort后可以方便找出重复的元素！   
        int ret1 = -1,ret2 = -1;
        int i = 0,size = nums.size();
        while(i < size - 1){
            if(nums[i] == nums[i+1])i = i+1+1;
            else {// 当nums[size-1]这个位置的元素是单个的时候，这个循环中无法判断这个位置的元素是否是单个的，因为直接跳过了！so再return之前应该再判断下这个位置是否已经把ret2填满了！
                if(ret1 == -1){
                    ret1 = nums[i];
                    i++;
                }else{
                    ret2 = nums[i];
                    i++;
                }
            }
        }
        // 处理一下末尾位置index的尴尬情况！
        if(nums[size-1] != nums[size-2])ret2 = nums[size-1];
        // 不论是 [1 1 4 6] or [1 1 2 3 3 4 4 10] 
        // 这2种case，都让i遍历到size-1这个位置就直接break出循环没法继续遍历了
        // 因此任何情况下其实nums[size-1]这个位置的元素并没有判断是否是不重复元素
        // 因此为这里需要再判断下！
        return {ret1,ret2};
    }
};
```

==LRU算法（大厂面试很喜欢出的一个题目！！！）：==

#### <9> [LeetCode 剑指 Offer II 031. 最近最少使用缓存](https://leetcode.cn/problems/OrIXps/)

这个思路为是跟着b站一个小up主学的
网站：https://www.bilibili.com/video/BV18A411i7ui?spm_id_from=333.337.search-card.all.click&vd_source=b050ab0adaffa51f5ad24d77efa40057

```c++
// 今晚我就把这个LRL算法彻彻底底学会即可了
// 因为这是个非常常见常考的面试题目！
// 先来默写个100遍！
#include<iostream>
#include<unordered_map>
using namespace std;
// 首先，LRU要求删除添加和查找的速度都快
// 因此就必须要结合双向list和哈希表这2种数据结果来do了！
int main(int argc, char* argv[]) {
	return 0;
}
// 首先，LRU要求删除添加和查找的速度都快
// 因此就必须要结合双向list和哈希表这2种数据结果来do了！

// 节点-结构体
struct Node {
	int _key, _val;   // 键 - 值
	Node* prev;		  // 指当前元素的 前一个元素 的指针 
	Node* next;		  // 指当前元素的 后一个元素 的指针 
	Node(int k, int v) // 构造函数
		:_key(k), _val(v), prev(nullptr), next(nullptr) {}
};
// 双向链表list（好do删除节点操作，若用单向list就麻烦很多，但也能do）
// 作用：存储数据
/*
	case1：正常存入（存入时容量未满）
	case2：存入的是重复的key
	case3：存入时容量满了
*/
struct DoubleList {
private:
	Node* head;					// 头结点处放入 最近刚刚使用过的 元素（头插法） 
	Node* tail;					// 尾结点处删除 最久没被使用过的 元素（尾删法）
public:
	DoubleList() {
		head = new Node(0, 0);
		tail = new Node(0, 0);
		head->next = tail;
		tail->prev = head;
	}
	~DoubleList() {
		if (head) {
			delete head; head = nullptr;
		}
		if (tail) {
			delete tail; tail = nullptr;
		}
	}
	void addFirst(Node* node) { // 在list头部插入元素
		node->next = head->next;
		node->prev = head;
		node->next->prev = node;
		head->next = node;
	} 
	int remove(Node* node) {// 容量满了之后直接删除最后一个节点即可
		int res = node->_key;
		node->next->prev = node->prev;
		node->prev->next = node->next;
		delete node; node = nullptr;
		return res;
	}	// 在list尾部删除元素
	int removeLastNode() {// 容量满了之后直接删除最后一个节点即可
		if (head->next == tail)return -1;
		return remove(tail->prev);
	}		
};
class LRUCache {
private:
	int _capacity;
	DoubleList* dlist;
	unordered_map<int, Node*> map;
public:
	LRUCache(int cap) {
		this->_capacity = cap;
		dlist = new DoubleList();
	}
	~LRUCache() {
		if (dlist) {
			delete dlist; dlist = nullptr;
		}
	}
	/* get方法分为2种case：
		1-map中	没有key对应的pair
		2-map中	有key对应的pair
	*/
	// get方法：获取key对应的value
	int get(int key) {
		auto it = map.find(key);
		if (it == map.end())return -1;	// map中没有
		// 有,就do 2个步骤：
		// 1-先将pair对的value拿出来，2-然后放到队列头部（因为最近使用过了）！
		int res = it->second->_val;
		put(key, res);			// 头插回去LRU中
		return res;
	}
	// put方法：将最近使用过的或者新的pair<key,val>插入到LRU的头部！
	void put(int key, int val) {
		// 找到
		// 找不到
		Node* tnode = new Node(key, val);
		auto it = map.find(key);
		if (it != map.end()) {// 找到有重复key时，直接删除，无需从尾部删除！
			dlist->remove(it->second);	// 先删除双向list中的该节点（）
			dlist->addFirst(tnode);		// 后再向双向list的头部添加该节点
			map[key] = tnode;			// 再重新do哈希表的映射关系
		}
		else {
			if (map.size() == this->_capacity) {// 因为满了so只能从尾部进行删除了！
				int nodeKey = dlist->removeLastNode();// 容量满了就直接删除最后一个元素
				// 然后把映射关系也删除！
				map.erase(nodeKey);// 按照key值把map中保存的映射关系给删除掉！
				// 然后再从头部添加进去！
				dlist->addFirst(tnode);
				map[key] = tnode;// 再do个哈希表的映射关系
			}
			else if (map.size() < this->_capacity) {
				dlist->addFirst(tnode);// 若没有达到容量,直接从头部添加
				map[key] = tnode;// 再do个哈希表的映射关系
			}
		}
	}
};
```



#### <10> [LeetCode 剑指 Offer 66.构建乘积数组](https://leetcode.cn/problems/gou-jian-cheng-ji-shu-zu-lcof/)

```c++
/*
    time:O(n * 2) == O(n)
    space:O(n),if不算res返回数组的话那就是O(1)的空间复杂度！
*/
class Solution {
public:
    vector<int> constructArr(vector<int>& a) {
        int size = a.size();
        vector<int> res(size,1);
        // 本题为通过学习评论区高赞答案：通过列表格法do出来！
        // 先计算左下三角，再计算右上三角的乘积和！
        /*
        举例子画图说明：   a = [1 2 3 4 5]
        1  1  <== ① 2 3 4 5 ==> 120 ==> 1x120(t == 1x5x4x3x2) == 120  
        2  1  <== 1 ① 3 4 5 ==> 60  ==> 1x60(t == 1x5x4x3)  == 60
        3  2  <== 1 2 ① 4 5 ==> 20  ==> 2x20(t == 1x5x4)  == 40
        4  6  <== 1 2 3 ① 5 ==> 5   ==> 6x5(t == 1x5)   == 30
        5  24 <== 1 2 3 4 ① ==> 1   ==> 24x1(t == 1)  == 24
        */
        for(int i = 1;i < size;++i){// 不管index==0号的值！因为已经定了！
            res[i] = res[i-1] * a[i-1];
        }
        int t = 1;
        for(int i = size-1;i >= 0;--i){// 不管index==size-1号的值！因为已经定了！
            // 这里从后往前来求右上三角度的乘积和太妙了吧！
            // 还有这下面的2行codes都非常棒！
            res[i] *= t;
            t      *= a[i];
        }
        return res;
    }
};
```



#### <11> [LeetCode 剑指 Offer 57.和为s的两个数字](https://leetcode.cn/problems/he-wei-sde-liang-ge-shu-zi-lcof/)

这道题目给我的启示：以后我自己刷题时一看到题目中==有已经排序过后数组==就要马上想起来用==双指针法！试一试看能不能deal题目！==

```c++
/*
    time:O(n)
    space:O(1)
*/
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        // 这是为看的评论区高赞大佬题解才知道能用《双指针》do出来的！
        // 这个题目意思原数组nums已经排序升序的数组了
        // 因此我就可以直接很容易就用 《双指针》 来deal的了！
        int l = 0,r = nums.size()-1;
        while(l < r){
            int tmp = nums[l]+nums[r];
            if(tmp < target)l++;// 太小了就l++让总和变大一点！
            else if(tmp > target)r--;// 太大了就r--让总和变小一点！
            else return {nums[l],nums[r]};// 刚好找到这个组合==target！
        }
        return {-1,-1};
    }
};
// way2:哈希表法
// time:O(n),最坏情况下需要遍历整个数组nums一遍！
// space:O(n),使用了辅助空间哈希表，最坏情况下遍历了整个nums数组都没有找到一对和为target的2个数字，此时哈希表就会存放nums数组的all元素！
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        // 一看就是要用哈希表来do！
        unordered_map<int,int> hashMap;
        int size = nums.size();
        for(int& num : nums){
            auto it = hashMap.find(target-num);
            if(it==hashMap.end()){
                hashMap[num]++;
            }else{
                return {it->first,num};
            }
        }
        return {-1,-1};
    }
};
```



#### <12> [LeetCode 剑指 Offer 57 - II.和为s的连续正数序列](https://leetcode.cn/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

这个题目的==第一种暴力解法==是纯纯我自己想的回溯法do出来的！（暴力回溯搜索法！so虽然在LC上能够通过，但是时间和空间复杂度非常地不咋地！）


```c++
// 方法1：暴力回溯+判断谁是连续序列的剪枝方法的结合 题解 我个人认为还是不错的！（PS:因为是我自己一个人想出来的哈~）
class Solution {
public:
    vector<vector<int>> res;
    vector<int> tmp;
    void backtracking(int sum,int tar,int startIdx){// 当然，一开始的sum序列和和为0！
        if(sum == tar){
            if(tmp.size() > 1)res.push_back(tmp);
            return;
        }
        for(int i = startIdx;i < tar;++i){
            // !tmp.empty() 这个条件前置是为了保证tmp.back()不出异常！
            if(!tmp.empty() && tmp.back() != i - 1)break;// 剪枝：非递增序列就无需暴力递归搜索求和了，因为本身就不符合题目的条件了！！！
            // 不是连续序列子数组也剪枝！这行代码就保证了结果数组是连续的！
            if(sum + i > tar)break;// 剪枝！ 
            sum += i;
            // 因为[1~target]的数组一定是升序的了，当前加起来超过target了
            // 那么后面的加起来也一定是会超过的！放心吧！
            tmp.push_back(i);
            backtracking(sum,tar,i+1);
            sum -= i;// 回溯
            tmp.pop_back();
        }
    }
    vector<vector<int>> findContinuousSequence(int target) {
        // 这个题目应该是需要用到暴力搜索回溯法！
        res.clear();tmp.clear();
        // 当然，一开始的sum序列和和为0！
        backtracking(0,target,1);// startIdx == 1是因为0不算入累加的行列中!
        // 用回溯法弄出来all从[1~target)数组中和为target的字数组后
        //（注：就算是一个数字tar == tar了也不满足题意！因为题目说了是至少要有2个元素的和==tar！）
        // 同时在暴力回溯的同时判断谁是严格升序子数组序列即可了！
        return res;
    }
};
```



#### <13> [LeetCode 剑指 Offer 58 - I.翻转单词顺序](https://leetcode.cn/problems/fan-zhuan-dan-ci-shun-xu-lcof/)

这个题目我之前2刷过carl哥的网站的题（lcT151），但是为忘记如何用双指针来do这个题目了！也就是说我虽然二刷了carl哥的除dp之外的题目，但是还是不能全覆盖地记住如何deal这些问题！

```c++
class Solution {
public:
    // 学carl哥之前教我的双指针法来do这套题！
    string reverseWords(string s) {
       	// 1-先去除原字符串前后的多余空格子
        // 2-再一个一个解析word 加入到反转字符串中 即可了！
        int l = 0,r = s.size()-1;
        while(l < s.size() && s[l] == ' ')l++;
        while(r >= 0 && s[r] == ' ')r--;
        int i = r;
        string ret = "";
        while(i >= l){// 注意：这里的遍历指针i一定是要求>=l,而不是>=0的,因为实际string的有效（非空格子起头or结尾）的部分是从s.substr[l,r]这部分的字符串！
            int j = i;
            while(j >= l && s[j] != ' ')j--;// 解析单个单词出来
            string word = s.substr(j+1,i-j);
            // 将解析好的单个单词输入到结果字符串中（因为是从后往前遍历原字符串，因此可以借此构造出新的原字符串的反向字符串了！）
            ret += word;
            if(j >= l)ret += " ";
            // 去掉s中间的多余空格子
            while(j >= l && s[j] == ' ')j--;
            i = j;// 更新i下标并do下一轮循环！
        }
        return ret;
    }
};
// 版本2：（从前往后，其实和上面的从后往前其实是一样的套路！）
// time:O(n)
// space:O(n),n是原字符串长度，使用了辅助空间stack存放了加起来长度共为n的words
class Solution {
public:
    string reverseWords(string s) {
        // 当然，因为题目给出的测试用例中的字符串包含前面或者后面有多个空余空格子的case，题目要求反转结果string中要去掉这些多余的空格子！
        int l = 0,r = s.size()-1;
        while(l<s.size() && s[l]==' ')l++;
        while(r>=0 && s[r]==' ')r--;
        int i = l,size = r+1;
        if(size == 0)return "";
        stack<string> stk;// 辅助栈空间
        while(i < size){
            int j = i;
            while(j < size && s[j] != ' ')j++;// 找每一个单词word
            string word = s.substr(i,j-i);
            stk.push(word);
            if(j < size)stk.push(" ");
            // 去掉string中间的多余空格子！
            while(j < size && s[j] == ' ')j++;
            i = j ;// 更新i do 下一轮反转word的操作！
        }
        // 把stack中存放到各个word和空格子弄进去结果string中！
        string ret = "";
        while(!stk.empty()){
            ret += stk.top();
            stk.pop();
        }
        return ret;
    }
};
```



#### <14> [LeetCode 剑指 Offer 61 .扑克牌中的顺子](https://leetcode.cn/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)

这个题目我是完全没有题解的思路，我自己的思路没法deal特殊case！不缺全面！很难想到这个题目的思路！真的！

```c++
// 这个顺子的题目总体的思路就是：
// 升序排序后比较最大最小牌值之差
// 小前提：构成顺子的这5个牌都必须是无重复元素的！
// if max - min < 5 ==> return true;
// else return false;

/*
   time:O(n),n是数组nums的大小
   space:O(1)
*/
class Solution {
public:
    bool isStraight(vector<int>& nums) {
        int size = nums.size(),minIdx = 0;
        std::sort(nums.begin(),nums.end());
        for(int i = 0;i < size - 1;++i){
            if(nums[i] == 0)minIdx++;// 找到最小牌的index
            else if(nums[i] == nums[i+1])return false;// 非0重复就不构成顺子，提取返回false
        }
        // max == nums[size-1].代表5个牌的最大值
        // min == nums[minIdx].代表5个牌的最小值（0值除外）
        return nums[size-1] - nums[minIdx] < 5;
        // 最大牌 - 最小牌 必须小于5才能构成顺子！
    }
};
```



#### <15> [LeetCode 剑指 Offer 64 .其1+2+...+n](https://leetcode.cn/problems/qiu-12n-lcof/)

这个题目我是完全没有想到不用if else while for switch case还有啥能do出来

然后看了题解才知道一种叫做==逻辑符号短路效应==的小知识点！

```c++
/*	本题用了著名的逻辑符号的短路效应：
	逻辑运算符的短路效应：
	if(A && B) 若 A 为 false ，则 B 的判断不会执行（即短路），直接判定 A && B 为 false
	if(A || B)若 A 为 true ，则 B 的判断不会执行（即短路），直接判定 A || B 为 true
	本题需要实现 “当 n=1时终止递归” 的递归终止条件这个需求，可通过 短路效应 来实现。
 	n > 1 && sumNums(n - 1) // 当 n = 1 时 n > 1 不成立 ，此时 “短路” ，终止后续递归
*/
class Solution {
public:
    int res = 0;
    int sumNums(int n) {
        // 因为不能用乘除法，还有各种if else while 和for，因此都不能用了！
        // 只能用递归不停地调用自己来累加！
        bool t = (n > 1) && sumNums(n-1);// 若n==1甚至小于1了，就无需再开多一个递归函数了！// 当n不>1时就不会继续do递归的操作了！so 短路效应牛逼！
        res += n;
        return res;
    }
};
```



#### <16> [剑指 Offer 67. 把字符串转换成整数](https://leetcode.cn/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof/)

这道题目其实我自己想的差不多了，就是差一点点判断越界的细节没想出来而已！

然后==参考了lc评论区的一个视频==，是B站up主https://www.bilibili.com/video/BV1mY411c7XJ/?vd_source=b050ab0adaffa51f5ad24d77efa40057的视频！然后才完全写出来的！

这道题目其实和[lc8. 字符串转换整数 (atoi)](https://leetcode.cn/problems/string-to-integer-atoi/)是一样的！复制黏贴过去也能AC！！！

```c++
// time:O(n),n是str的长度
// space:O(10)==O(1),使用了辅助哈希表空间，但是只用了10个常数级空间，so和O(1)是一回事！
class Solution {
public:
    unordered_map<char,int> hashMap;
    int strToInt(string str) {
        hashMap['0']++;hashMap['1']++;hashMap['2']++;hashMap['3']++;hashMap['4']++;
        hashMap['5']++;hashMap['6']++;hashMap['7']++;hashMap['8']++;hashMap['9']++;
        int size = str.size();
        if(size == 0)return 0;
        int idx = 0;// 用来do遍历的索引
        int PorN = 1;// 1表示正数 -1表示负数
        // 先去除字符串str前面多余的空格子！
        while(idx < size && str[idx] == ' ')idx++;
        // 判断特殊case + 符号判断（判断第一个非空字符的正负性）
        if(idx < size && hashMap.find(str[idx]) == hashMap.end()){
            if(str[idx] == '-')PorN = -1;
            else if(str[idx] == '+')PorN = 1;
            else return 0;// 在数字前面若出现了非数字字符那这个题目默认就是返回0的
            idx++;// 跳到str中真正以数字字符开头的下标处！
        }
        // 若第一个非空字符不是+ or -号，则直接跳过上面的if语句直接到下面的循环中计算数字大小了！
        long long res = 0;
        // 用long long作为res的类型是防止lc后台测试用例数字过大导致移除错误！
        for(;idx < size;++idx){
            int digit = str[idx] - '0';// 将每一个数字字符转换为对应的int数字(-'0'技巧)
            if(digit < 0 || digit > 9)break;// str的中间存在非法字符，不能继续do循环计算数字和的处理了！
            res = res * 10 + digit;// 计数
            if(res * PorN >= INT_MAX)return INT_MAX;
            if(res * PorN <= INT_MIN)return INT_MIN;
        }
        return (int)res * PorN;
    }
};
```



#### <17> [剑指 Offer 42. 连续子数组的最大和](https://leetcode.cn/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

这个题目和[lc53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/)是一毛一样的解题思路！我之前在carl哥那do过2次了，但是这次还是没有记住怎么do！

```c++
// time:O(n),n是原数组nums的大小！
// space:O(1)
// 贪心算法，非常地巧妙！！！
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int res = INT_MIN,cnt = 0,size = nums.size();
        // res保存结果,cnt保存临时连续子数组和！
        for(int i = 0;i < size;++i){
            cnt += nums[i];
            res = res < cnt ? cnt:res;// 保存最大连续子数组和
            if(cnt < 0)cnt = 0;// 若cnt小于0了，就不符合连续子数组构成的最大和这个条件了
            // 就让cnt重新赋值为0，继续求连续和的最大值！
        }
        return res;
    }
};
```



#### <18> [剑指 Offer 33. 二叉搜索树的后序遍历序列](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)

本题我是看了评论区“递归和栈两种方式解决，最好的击败了100%”的用户这个标题的大佬才do出来的！

https://leetcode.cn/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/solution/di-gui-he-zhan-liang-chong-fang-shi-jie-jue-zui-ha/

```c++
// 分治递归法：(左闭右闭区间)
// 思路：对于BST来说，后序遍历后最后一个元素必然是根节点root
// 然后后续遍历的第一个大于root的节点到root的前一个节点这个区间内的节点必然为BST的右子树
// 然后从前面到第一个大于root的节点的前一个节点的区间必然是BST的左子树
// 即：[left,第一个大于当前root节点-1]是BST的左子树区间
//    [第一个大于当前root节点,root节点-1]是BST的右子树区间
// 若不满足以上的2个条件的话，就一定不是BST了！
class Solution {
public:
    bool traversal(const vector<int>& p,int l,int r){
        if(l >= r)return true;
        int fenGeIdx = l;
        int curRoot = p[r];// 因为BST的后续遍历最后一个节点值肯定是当前根节点的值！so直接取p[r]==root
        while(p[fenGeIdx] < curRoot)fenGeIdx++;
        /*
            因为要确认当前的[第一个大于当前root节点,root节点-1]区间必须都要大于curRoot值
            才能确定是BST当前根节点curRoot的右子树！
            若有小于curRoot值的就马上return false表示不是BST了！
        */
        int tmp = fenGeIdx;
        while(tmp < r){// 右子树但凡有小于root的马上判断不是BST
            if(p[tmp++] < curRoot)return false;
        }
        /*
            同上理，要确定[left,第一个大于当前root节点-1]是BST的左子树区间这个区间的all节点值
            都要小于当前根节点curRoot的值！才符合BST！
        */
        tmp = fenGeIdx - 1;
        while(tmp >= l){// 左子树但凡有大于root的马上判断不是BST
            if(p[tmp--] > curRoot)return false;
        }
        // 判断左子树是否为BST
        bool leftIsBST = traversal(p,l,fenGeIdx-1);
        // 判断右子树是否为BST
        bool rightIsBST = traversal(p,fenGeIdx,r-1);
        // 进而判断整棵树是否为BST
        return leftIsBST && rightIsBST;
    }
    bool verifyPostorder(vector<int>& postorder) {
        return traversal(postorder,0,postorder.size()-1);
    }
};
// 我自己写出来的
class Solution {
public:
    bool traversal(const vector<int>& p,int l,int r){
        // 因为BST的后续遍历到最后一个节点必须是当前树的根节点！
        // 而[start,第一个大于root节点的前一个节点]必须是BST的左区间
        // [第一个大于root节点,root节点的前一个节点]必须是BST的右区间
        // 根据这2个规则就能递归判断出这个序列是否是某个BST的后续遍历序列了！
        if(l >= r)return true;// 剩下一个数字肯定符合BST的左右区间的定义了！
        int i = l;
        while(p[i] < p[r])i++;//跳出循环后==> [l,i-1]是BST的左区间
        int tmp = i;
        while(p[tmp] > p[r])tmp++;//跳出循环后==> [i,r-1]是BST的右区间
        if(tmp != r)return false;
        bool left = traversal(p,l,i-1);
        bool right = traversal(p,i,r-1);
        return left && right;
    }
    bool verifyPostorder(vector<int>& postorder) {
        return traversal(postorder,0,postorder.size()-1);
    }
};
```



#### <19> [剑指 Offer 34. 二叉树中和为某一值的路径](https://leetcode.cn/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/)

##### [113. 路径总和 II](https://leetcode.cn/problems/path-sum-ii/)

这两个题目是一毛一样的呀！我自己还是在carl哥的代码随想录的二叉树系列刷了很多次了的！！！还是没法完整写出来！

```c++
class Solution {
public:
    // 分析：因为我们此时要找出所有从根节点到叶子节点路径总和==给定目标和的路径
    // 因此我们必须要遍历整一颗二叉树，因此递归函数不需要返回值
    // 用递归函数从前到后用前序遍历的思路来do就符合找根节点到叶子节点的路径的思路了！
    // 这个思路来自carl哥的代码随想录！
    void traversal(TreeNode* cur,vector<int>& tpath,vector<vector<int>>& path,int tar){
        // 整体上是用到前序遍历 :中 左 右的logic来遍历二叉树的！
        if(!cur->left && !cur->right){
            // 遍历到根节点才算一个有效路径
            // 然后再看看是否符合路径和==目标和！
            if(tar == 0){
                path.push_back(tpath);
                return;// 符合目标和就加入结构路径path中，并继续遍历其他路径
            }else return;// 不符合，就继续回溯遍历其他路径
        }
        if(cur->left){
            // 处理当前节点
            tar -= cur->left->val;
            tpath.push_back(cur->left->val);
            // 继续递归
            traversal(cur->left,tpath,path,tar);
            // 回溯
            tar += cur->left->val;
            tpath.pop_back();
        }
        if(cur->right){
            // 处理当前节点
            tar -= cur->right->val;
            tpath.push_back(cur->right->val);
            // 继续递归
            traversal(cur->right,tpath,path,tar);
            // 回溯
            tar += cur->right->val;
            tpath.pop_back();
        }
    }
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        if(root == nullptr)return {};
        vector<int> tpath;
        vector<vector<int>> path;
        tpath.push_back(root->val);// 不论如何你的整棵树的根节点root一定是先要加入到路径中的
        // 因为你题目说了，根节点到叶子节点的路径才算完整de有效路径！
        traversal(root,tpath,path,targetSum - root->val);
        return path;
    }
};
```



#### <20> [剑指 Offer 30. 包含min函数的栈](https://leetcode.cn/problems/bao-han-minhan-shu-de-zhan-lcof/)

##### [155. 最小栈](https://leetcode.cn/problems/min-stack/)

又是一个我在carl哥那刷过很多次的题目，我没有再次写出来！

```c++
class MinStack {
public:
    /** initialize your data structure here. */
    stack<pair<int,int>> stk;
    // stk存储的是一个队组，first表示元素值，second表示当前栈中存储的元素值中的最小值！
    MinStack() {
        
    }
    void push(int x) {
        if(stk.empty()){
            stk.push(pair<int,int>(x,x));
        }else{
            stk.push(pair<int,int>(x,std::min(x,stk.top().second)));
            // 因为与本结构中的成员函数min有重名，因此要加上std::调用标准的名字空间中的min来do区分！
        }
    }
    void pop() {
        if(stk.empty())return;
        stk.pop();
    }
    int top() {
        // 要是只用一个普通的stack<int>栈的话就不能正确弄出原本加入到top值！
        if(stk.empty())return -1;
        return stk.top().first;
    }
    int min() {
        if(stk.empty())return -1;
        return stk.top().second; 
    }
};
```



#### <21> [剑指 Offer 68 - II. 二叉树的最近公共祖先](https://leetcode.cn/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

##### [236. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/)

这又是一道我在carl哥那刷过的题目，但是我忘记怎么做了！

```c++
class Solution {
public:
    /* 
    思路：求BT的最近公共祖先，就必须要想到要用回溯法！也即后续遍历BT的logic
        从下往上遍历！当 当前节点是空or p or q时直接返回cur
        然后遍历其左右子树
        当左右都不空，说明当前节点就是最近公共祖先，直接返回即可
        当左空右不空，说明最近公共祖先节点会由右子树直接返回
        当左不空右空，说明最近公共祖先节点会由左子树直接返回
        当左右都为空，直接返回左or右，此时没有在以当前cur节点为根节点的树的左右子树中找到公共祖先节点
        so直接返回nullptr
    */
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root == nullptr || root == q || root == p)return root;
        TreeNode* left = lowestCommonAncestor(root->left,p,q);
        TreeNode* right = lowestCommonAncestor(root->right,p,q);
        if(left && right)return root;
        else if(left && !right)return left;
        else if(!left && right)return right;
        return right; 
    }
};
```



#### <22> [剑指 Offer 68 - I. 二叉搜索树的最近公共祖先](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

##### [235. 二叉搜索树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-search-tree/)

这还是一道我在carl哥那刷过的题目，但是我忘记怎么做了！

```c++
// way1：迭代法求BST的最近公共祖先节点！
// 这里利用了BST的节点的值的有序性！左->val < 根->val < 右->val <==> 根是左和右的最近公共祖先节点了！
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root == nullptr)return root;
        while(root){
            if(root->val < p->val && root->val < q->val){
                root = root->right;// 去右边找最近公共祖先节点！
            }else if(root->val > p->val && root->val > q->val){
                root = root->left;// 去左边找最近公共祖先节点！
            }else return root;// 当前节点就是p和q的最近公共祖先节点了！
        }
        return nullptr;// 迭代找不到最近公共祖先节点的话就直接返回空nullptr即可！
    }
};
// way2：当然你也可以把BST当作是普通二叉树那样，用递归来求最近公共祖先节点！
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root == nullptr || root == q || root == p)return root;// 空节点的最近公共祖先肯定也是空节点！    
        TreeNode* left = lowestCommonAncestor(root->left,p,q);
        TreeNode* right = lowestCommonAncestor(root->right,p,q);
        if(left && right)return root;
        else if(left && !right)return left;
        else if(!left && right)return right;
        return left;
    }
};
```



#### <23> [剑指 Offer 26. 树的子结构](https://leetcode.cn/problems/shu-de-zi-jie-gou-lcof/)

*学的这个lc题目评论区的高赞答案写的！！！太NB了！可太难想了这个思路！*

==思路：==

1- 先在树A中找到头结点 == 树B头结点的节点

2- 找到之后再将此时的A子结构的left和right 与 B的left和right逐个do队比，即可判断A是否含有B这样的子结构了！

```c++
class Solution {
public:
    bool isSubStructure(TreeNode* A, TreeNode* B) {
        if(A == nullptr || B == nullptr)return false;// 空子树不是任何一棵树的子结构
        if(A->val == B->val){// 只有找到同一个根节点时才能接下去判断是否匹配的问题！
            if(traversal(A,B))return true;
        }
        // A的左orright 只有含有B这样的结构就都算是true
        return isSubStructure(A->left,B) || isSubStructure(A->right,B);
    }
    // 下面这个子函数是判断这2个可能的子结构A和B到底是否是匹配的
    bool traversal(TreeNode* A,TreeNode* B){
        if(B == nullptr)return true;// 若判断出B遍历完成了，那肯定子结构A与B相互匹配了！
        if(A == nullptr || A->val != B->val)return false;
        // 若判断出A遍历完成了(到A的叶子节点了，再往该节点的左右遍历都是空了)
        // or 说遍历到当前A的子结构该节点的值！=B对应节点的值，那肯定子结构A与B相互不匹配了！
        // 此时就是A子结构的当前节点和B的对应节点值相等的case，就继续递归遍历其他节点的值
        // 看是否匹配！
        return traversal(A->left,B->left) && traversal(A->right,B->right);
    }
};
```



#### <24> [剑指 Offer 16. 数值的整数次方](https://leetcode.cn/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/)

##### [50. Pow(x, n)](https://leetcode.cn/problems/powx-n/)这个lc50也同上！是同一个题目！

本题的方法被称为==「快速幂算法」==!我学自lc官方题解区的官方视频题解！！！忘了就回看一下官方给出的视频讲解即可！https://leetcode.cn/problems/powx-n/solution/powx-n-by-leetcode-solution/

==快速幂次算法==本质就是==分治算法==，时间复杂度就是O（logn）的！

思路：

分治算法：（通过下面这2个详细例子你自己也能够悟出来的了！）

x ^16 <=> x^16 -> x^8 * x^8 -> x^4 * x^4 * x^4 * x^4  ->  x^2 * x^2 * x^2 * x^2 * x^2 * x^2 * x^2 * x^2

x ^20 <=> x^20 -> x^10 * x^10-> x^5 * x^5 * x^5 * x^5 ->  x^2 * x^2 * x *x^2 * x^2 * x

然后思路就是递归分治！（若是理解了上面这2个例子要写出递归分治的codes不难了！）

当n < 0时，取result == 1.0 / pow(x,abs(n));

当n >= 0时，取result == pow(x,n);

```c++
// 时间：O(logn)，递归的层数
// 空间：O(logn)，递归的层数，这是由于递归的函数调用会使用栈空间。
class Solution {
public:
    // 其实这个代码时有问题的，因为没有返回值！
    double myPowHelper(double x, uint32_t n) {// 在leetcode上面写这个题目的话不需要谢伟uint32_t,但是牛客就要改一下！
        // uint32_t来表示n才能不越界！NB！！！当然你甚至可以用uint64_t也ok！
        if(n == 0)return 1.0;// 当n递减到0时就代表x到最后了！就直接x个1.0完事了！
        if(n % 2 != 0){
            double half = myPowHelper(x,n / 2);
            return half * half * x;
        }else{
            double half = myPowHelper(x,n / 2);
            return half * half;
        }
    }
    // 其实这个代码才算完全没问题的！
    double dfs(double x,int n){
        if(n == 0)return 1.0;// 任何数的0次方==1.0！
        double half = dfs(x,n / 2);
        if(n % 2 != 0)return half * half * x; 
        return half * half;// else if(n % 2 == 0)return half * half; 
    }
    double myPow(double x,int n){
        // 必须要考虑到n == 0 和 x == 1时这种特殊case！
        if(n == 0 || x == 1)return 1;
        // 任何数的0次方 or 1的任何次方都 == 1！
        if(n < 0){
            return 1.0 / myPowHelper(x,abs(n));
        }
        return myPowHelper(x,n);
    }
};
```

#### <25> [剑指 Offer 07. 重建二叉树](https://leetcode.cn/problems/zhong-jian-er-cha-shu-lcof/)

这个题目我在carl哥那也刷了N遍了，但是还是不会do！那就背下来把！！！

```c++
 /*
    思路：先（每次递归都用前序的第一个元素来）切割中序，后（每次递归都用中序左的size来）切割前序列
    		[3 9 20 15 7]
            [9 3 15 20 7]
            /           \ 
           [9]          [20 15 7]    
           [9]          [15 20 7]
                       /        \
                     [15]     [7]
                     [15]     [7]
 */
class Solution {
public:
    TreeNode* traversal(vector<int>& preorder, vector<int>& inorder) {
        int size = preorder.size();
        if(size == 0)return nullptr;
        TreeNode* root = new TreeNode(preorder[0]);
        if(size == 1)return root;
        // 先切割中序(按照左开右闭原则)
        int qiegeIdx = -1;
        for(int i = 0;i < inorder.size();++i){
            if(inorder[i] == preorder[0]){
                qiegeIdx = i;break;
            }
        }
        vector<int> leftInorder(inorder.begin(),inorder.begin()+qiegeIdx);
        vector<int> rightInorder(inorder.begin()+qiegeIdx+1,inorder.end());
        // 后切割前序(按照左开右闭原则)
        preorder.erase(preorder.begin(),preorder.begin()+1);// 先物理删除用过的前序第一个元素值（当前根节点值）
        int leftInorderSize = leftInorder.size();
        vector<int> leftPreorder(preorder.begin(),preorder.begin()+leftInorderSize);
        vector<int> rightPreorder(preorder.begin()+leftInorderSize,preorder.end());
        // 用递归函数返回值do重新构造二叉树的工作！
        root->left = traversal(leftPreorder,leftInorder);// 继续构造左子树
        root->right = traversal(rightPreorder,rightInorder);// 继续构造右子树
        return root;
    }
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        if(preorder.size() == 0 || inorder.size() == 0)return nullptr;
        return traversal(preorder,inorder); 
    }
};
```



下面这个题目与上面的思路类似：

##### [106. 从中序与后序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

```c++
 /*
    思路：先（每次递归都用后序的last一个元素来）切割中序，后（每次递归都用中序左的size来）切割后序列
            [9 3 15 20 7]
            [9 15 7 20 3]
            /           \ 
           [9]          [15 20 7]    
           [9]          [15 7 20]
                       /        \
                     [15]      [7]
                     [15]      [7]
 */
class Solution {
public:
    TreeNode* traversal(vector<int>& inorder, vector<int>& postorder){
        int size = postorder.size();
        if(size == 0)return nullptr;
        TreeNode* root = new TreeNode(postorder[size-1]);
        if(size == 1)return root;
        // 若当前序列数组只有一个元素，就直接返回把！不用继续递归构建BT了！
        // 先切割中序
        // 1:找切割中序的切割点
        int qiegeIdx = -1;
        for(int i = 0;i < inorder.size();++i){
            if(inorder[i] == root->val){
                qiegeIdx = i;break;
            }
        }
        // 2:切割中序(以左闭右开原则)
        vector<int> leftInorder(inorder.begin(),inorder.begin()+qiegeIdx);
        vector<int> rightInorder(inorder.begin()+qiegeIdx+1,inorder.end());
        // 后切割后序(以左闭右开原则)
        postorder.resize(postorder.size()-1);// 物理上pop掉最后一个用过的元素值（是当前根节点）
        vector<int> leftPostorder(postorder.begin(),postorder.begin()+leftInorder.size());
        vector<int> rightPostorder(postorder.begin()+leftInorder.size(),postorder.end());
        // 利用递归函数返回值do重新构造二叉树的工作
        root->left = traversal(leftInorder,leftPostorder);
        root->right = traversal(rightInorder,rightPostorder);
        return root; 
    }
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        if(inorder.size() == 0 || postorder.size() == 0)return nullptr;
        return traversal(inorder,postorder);
    }
};
```



#### <26> [剑指 Offer 43. 1～n 整数中 1 出现的次数](https://leetcode.cn/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/)

这个困难题目就当让我长长见识得了！太难了！说实话面试要是遇到了那也只能G了！

```c++
// 这个题目我是看的b站up主香辣鸡排蛋包饭才知道这么do的！
// 但是这也太难了吧,我说实话，只能把这种思路完全记忆背下来才能默写出来！
/*
    time:O(1),就算是百万位的数字，也只有需要遍历7次，因为只有7位的数字！O(7) == O(1)
    space:O(1)
*/
class Solution {
   public:
    int countDigitOne(int n) {
        int res = 0;// 保存结果
        long base = 1;// 从个位开始统计
        // 对于数字n，要 求从1~n的all数字中出现'1'的次数 == 求sum(各个位数上面出现的'1'的次数)
        while(base <= n){
            int a = n / base / 10;
            int b = n % base;
            int cur = n / base % 10;
            if(cur > 1){
                res += (a+1)*base;
            }else if(cur == 1)res += a*base + b + 1;
            else res += a*base;
            base *= 10;// 依次统计各个位置上到‘1’出现的次数！
        }
        return res;
    }
};
```



#### <27> [剑指 Offer 51. 数组中的逆序对](https://leetcode.cn/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

这道题目虽然是困难级别的，但还是能做的，不像上面<26>题那样变态！！！虽说也挺难想的，但是我做这个题目的同时也复习了一下归并排序如何写！so这个题目还是值得一做的！

==这道题目主要参考的是B站up主：香辣鸡排蛋包饭的video讲解！==

https://www.bilibili.com/video/BV1CK411c7gx?p=45&vd_source=b050ab0adaffa51f5ad24d77efa40057

还有一篇lc题解区的小题解：

https://leetcode.cn/problems/shu-zu-zhong-de-ni-xu-dui-lcof/solution/jian-zhi-offer-51-shu-zu-zhong-de-ni-xu-a80cq/

```c++
/*
    考察的知识点：归并排序模板默写的同时，统计逆序对！
    归并排序会将数组各个元素值do升序排序（默认），同时我们再统计一下逆序对即可over 本题了！
    当有 ：if(nums[l_pos] > nums[r_pos])时，马上让 res += (mid - l_pos + 1);即可统计逆序对了！
*/
class Solution {
private:
    int res = 0;
public:
    int reversePairs(vector<int>& nums) {
        //使用归并排序法来do这道题目！
        // 求逆序对数，则必须是使用 归并排序 的过程！
		// 先递归分割，后递归合并的同时计算逆序对数！！！
        vector<int> tmp;
        int size = nums.size();
        tmp.resize(size);// 必须要提前resize，否则lc就是会给你判错！
        if(size == 0 || size == 1)return 0;// 特殊case，此时没有逆序对
        mergeSort(nums,0,size-1,tmp);
        return res;
    }
    void mergeSort(vector<int>& nums,int left,int right,vector<int>& tmp){
        if(left >= right)return;// 一个元素本身就是有序的了！
        int mid = left + (right - left) / 2;
        // 递归分割左区间
        mergeSort(nums,left,mid,tmp);
        // 递归分割右区间
        mergeSort(nums,mid+1,right,tmp);
        // 合并,同时统计逆序对的个数res
        merge(nums,left,mid,right,tmp);
    }
    void merge(vector<int>& nums,int left,int mid,int right,vector<int>& tmp){
        int l_pos = left,r_pos = mid+1,pos = left;
        while(l_pos <= mid && r_pos <= right){
            if(nums[l_pos] <= nums[r_pos])tmp[pos++] = nums[l_pos++];
            else {
                res += (mid - l_pos+1);// 在Cpp版本的归并排序模板中就写这一句代码就能通过此题！
                tmp[pos++] = nums[r_pos++];
            }
        }
        while(l_pos <= mid)tmp[pos++] = nums[l_pos++];
        while(r_pos <= right)tmp[pos++] = nums[r_pos++];
        for(int i = left;i <= right;++i)nums[i] = tmp[i];
    }
};
```

补充：==归并排序模板==

归并排序B站大佬讲解的video：https://www.bilibili.com/video/BV1Pt4y197VZ?spm_id_from=333.337.search-card.all.click&vd_source=b050ab0adaffa51f5ad24d77efa40057

```c++
// 1-Cpp版归并排序模板：
#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<vector>
#include<algorithm>
using namespace std;
void merge(vector<int>& nums,int left,int mid,int right,vector<int>& tmp){
    int l_pos = left,r_pos = mid+1,pos = left;
    while(l_pos <= mid && r_pos <= right){
        if(nums[l_pos] <= nums[r_pos]){
            tmp[pos++] = nums[l_pos++];
        }
        else {
            tmp[pos++] = nums[r_pos++];
        }
    }
    while(l_pos <= mid)tmp[pos++] = nums[l_pos++];
    while(r_pos <= right)tmp[pos++] = nums[r_pos++];
    for(int i = left;i <= right;++i)nums[i] = tmp[i];
}
void mergeSort(vector<int>& nums,int left,int right,vector<int>& tmp){
    if(left >= right)return;// 一个元素本身就是有序的了！
    int mid = left + (right - left) / 2;
    // 递归分割左区间
    mergeSort(nums,left,mid,tmp);
    // 递归分割右区间
    mergeSort(nums,mid+1,right,tmp);
    // 合并
    merge(nums,left,mid,right,tmp);
}
void print(const vector<int>& nums){
    for_each(nums.begin(),nums.end(),[](int v){cout<<v<<"\t";});
    cout<<endl;
}
int main(int argc,char* argv[]){
    vector<int> nums{9,5,2,7,12,4,3,1,11};// {7,5,6,4};
    vector<int> tmp;
    tmp.resize(nums.size());
    print(nums);
    mergeSort(nums,0,nums.size()-1,tmp);
    print(nums);
    return 0;
}
/*--------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
// 2-C语言版归并排序模板：
#include<stdio.h>
#include<stdlib.h>
void print(int * arr,int n){
    for(int i = 0;i < n;++i){
        printf("%d\t",arr[i]);
    }
    printf("\n");
}
// 合并函数
void merge(int* arr,int* tmpArr,int left,int mid,int right)
{
    // 标记左半区第一个未排序的元素de index
    int l_pos = left;
    // 标记右半区第一个未排序的元素de index
    int r_pos = mid + 1;
    // 临时数组元素的下标
    int pos = left;
    // 合并
    while(l_pos <= mid && r_pos <= right){
        if(arr[l_pos] < arr[r_pos]){
            tmpArr[pos++] = arr[l_pos++];
        }else{
            tmpArr[pos++] = arr[r_pos++];
        }
    }
    // 合并左半区剩余的元素
    while(l_pos <= mid){
        tmpArr[pos++] = arr[l_pos++];
    }
    // or合并右半区剩余的元素
    while(r_pos <= right){
        tmpArr[pos++] = arr[r_pos++];
    }
    // 把临时数组中合并后的元素复制回原来的数组
    for(int i = left;i <= right;++i){
        arr[i] = tmpArr[i];
    }
}
// 归并排序的实质函数
void msort(int * arr,int * tmpArr,int left,int right){
    // 如果只有一个元素的区域就不需要继续划分了
    // 只需要被归并，因为一个元素的区域本身就是有序的了！无需排序了！
    if(left >= right)return;
    // 找中间点
    int mid = left + (right - left) / 2;
    // 递归划分左半区域
    msort(arr,tmpArr,left,mid);
    // 递归划分右半区域
    msort(arr,tmpArr,mid+1,right);
    // 合并已经排好序的部分区域
    merge(arr,tmpArr,left,mid,right);
}
// 归并排序的入口函数
void merge_sort(int * arr,int n){
    int* tmpArr = (int*)malloc(n * sizeof(int));
    if(tmpArr){
        msort(arr,tmpArr,0,n-1);
        free(tmpArr);
    }else{
        printf("error,failed to allocate tmp memory!\n");
    }
}
int main(int argc,char* argv[]){
    int arr[] {9,5,2,7,12,4,3,1,11};
    int n = 9;
    print(arr,n);
    // 归并算法总体思路：先递归切分然后再排序合并即可！
    merge_sort(arr,n);
    print(arr,n);
    return 0;
}
```

#### <28> [剑指 Offer 45. 把数组排成最小的数](https://leetcode.cn/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)

这个题目的题解是非常巧妙的！！！利用了==字符串相加后再比大小==的技巧！还是从lc题解区的路飞k神大佬那学的！！！这个技巧还是能长长见识，然后记忆下来的！

<img src="F:\学习过程中的截图\lc剑指offer45.png" alt="lc剑指offer45" style="zoom: 67%;" />

```c++
struct myPred:public binary_function<string,string,bool>{
    bool operator()(const string& x,const string& y)const{
        return (x + y) < (y + x);// 拼接成输出最小的数字字符串的自定义排序谓词！
    }
};
class Solution {
public:
    string minNumber(vector<int>& nums) {
        vector<string> strs;
        for(int& num : nums){// 先将每个数字转换为字符串，并加入到字符串数组中去！
            strs.push_back(to_string(num));
        }
        // 按照// 自定义排序拼接成输出最小的数字！
        std::sort(strs.begin(),strs.end(),myPred());
        string res = "";
        for(auto str:strs){// 弄成一个字符串好do return
            res += str;
        }
        return res;
    }
};
```



#### <29> [剑指 Offer 48. 最长不含重复字符的子字符串](https://leetcode.cn/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

这个题目和[3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)是一样的！

```c++
// 滑动窗口（双指针法）,这是学的y总的！
// 得自己按照lc给出的特例，比如第一个特例，拿来跟着我这份代码画个图就行了！
// time:O(n),n是原字符串的长度，1个for循环遍历原字符串的时间复杂度就是O(n)
// space:O(n),n是原字符串的长度，最坏情况下，uset需要保存n个字符！
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int res = 0,size = s.size();
        unordered_set<char> unset;
        for(int i = 0,j = 0;i < size;++i){
            // 只要发现是重复元素就继续删除掉set中的对应元素
            while(unset.find(s[i]) != unset.end()){
                unset.erase(s[j++]);
            }
            unset.insert(s[i]);// 继续加入新元素
            res = max(res,i-j+1);// 更新最长不重复数组长度！
        }
        return res;
    }
};
```





## **牛客网  《面试必刷TOP101 》刷题笔记**

这里的题目都是我自己一刷时没do出来的（看题解） or do的有些磕磕绊绊需要想挺久才do出来的题目！！！（暗示我自己这些剑指offer的题目为肯定是需要2刷甚至3or4刷的！）

我到时候面试的时候就一定要先把==10大排序算法==再背回来！清除知道了解他们的时间空间复杂度！==(這一點是我在刷lc题目的时候没注意到的！)==

注意：**我个人认为太过于难的题目我就不总结了！**

### 目前累计总共有==《54》==道题：

```c++
// 注意：
// 对于动态规划的题目，一定要记住carl哥的dp5部曲中的4部曲！！！
// 1-给出明确的dp数组的下标含义
// 2-给出dp数组的初始化
// 3-确定dp数组的遍历顺序
// 4-确定递归公式
```





#### <1> [BM2-链表内指定区间反转](https://www.nowcoder.com/practice/b58434e200a648c589ca2063f1faf58c?tpId=295&tqId=654&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

<img src="C:/Users/11602/Desktop/LeetCodeOfferNotes/git/%E7%AE%97%E6%B3%95%E5%AD%A6%E4%B9%A0%E6%88%AA%E5%9B%BE/1.jpg" alt="1" style="zoom: 50%;" />

```c++
/**
 * struct ListNode {
 *	int val;
 *	struct ListNode *next;
 * };
 */
// time:O(n),最坏情况下，需要遍历整个list，比如m or n 是指向最后一个节点时
// spcae:O(1),占用了常量级别的指针内存空间
class Solution {
public:
    // 我说实话，这个题目我看了官方题解之后真的觉得很nb的！（画个图就出来了！）
    ListNode* reverseBetween(ListNode* head, int m, int n) {
        ListNode* dummyNode = new ListNode(-1);// 虚拟头结点
        dummyNode->next = head;
        ListNode* prev = dummyNode;
        ListNode* cur = head;
        // 找到m这个位置的节点
        for(int i = 1;i < m;++i){
            prev = cur;
            cur = cur->next;
        }
        // 给从m到n位置的all节点do反转！
        for(int i = m;i < n;++i){
            ListNode* tmp = cur->next;
            cur->next = tmp->next;
            tmp->next = prev->next;
            prev->next = tmp;
        }
        return dummyNode->next;// 返回真正的list头结点！
    }
};
```



#### <2> [BM3-链表中的节点每k个一组翻转](https://www.nowcoder.com/practice/b49c3dc907814e9bbfa8437c251b028e?tpId=295&tqId=722&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

```c++
/**
 * struct ListNode {
 *	int val;
 *	struct ListNode *next;
 * };
 */
// time:O(n),最坏情况下需要遍历整个list，space:O(1),只占用了常量指针内存空间！
class Solution {
public:
    // 思路：每k个节点do一次部分反转list的操作即可！
    // 		运用到了属于BM2中部分反转list的操作！(个人认为我自己基于BM2这个题解写的思路比BM3要好！)
    ListNode* reverseKGroup(ListNode* head, int k) {
        int size = 0;
        ListNode* cur = head;
        // 求list size
        while(cur){
            cur = cur->next;
            size++;
        }
        // 处理下特殊case：size == k 时应该让list全反转
        if(size == k){
            head = reverseAll(head);
            return head;
        }
        // size != k 时应该让list部分反转
        int i = 1;
        while(1){
            if( (i+k-1) >= size)break;
            head = subReverseFun(head,i,i+k-1);
            i += k;
        }
        return head;
    }
    ListNode* subReverseFun(ListNode* head,int m,int n){
        ListNode* dummyNode = new ListNode(-1);
        dummyNode->next = head;
        ListNode* prev = dummyNode;
        ListNode* cur = head;
        // 找到m位置的节点
        for(int i = 1;i < m;++i){
            prev = cur;
            cur = cur->next;
        }
        // 从m反转到n
        for(int i=m;i<n;++i){
            ListNode* tmp = cur->next;
            cur->next = tmp->next;
            tmp->next = prev->next;
            prev->next = tmp;
        }
        return dummyNode->next;
    }
    // list全反转
    ListNode* reverseAll(ListNode* head){
        ListNode* prev = nullptr;
        ListNode* cur = head;
        while(cur){
            ListNode* tmp = cur->next;
            cur->next = prev;
            prev = cur;
            cur = tmp;
        }
        return prev;
    }
};
// version-2
class Solution {
public:
    // 思路：每k个节点do一次部分反转list的操作即可！
    // 		运用到了属于BM2中部分反转list的操作！(个人认为我自己基于BM2这个题解写的思路比BM3要好！)
    ListNode* reverseKGroup(ListNode* head, int k) {
        int size = 0;
        ListNode* cur = head;
        // 求list size
        while(cur){
            cur = cur->next;
            size++;
        }
        // 处理下特殊case：size == k 时应该让list全反转
        if(size == k){
            head = subReverseFun(head,1,size);// list全反转
            return head;
        }
        // size != k 时应该让list部分反转
        int i = 1;
        while(1){// 从1开始算，让区间中的每k个节点的区间中的节点进行反转！
            if( (i+k-1) >= size)break;
            head = subReverseFun(head,i,i+k-1);
            i += k;
        }
        return head;
    }
    ListNode* subReverseFun(ListNode* head,int m,int n){
        ListNode* dummyNode = new ListNode(-1);
        dummyNode->next = head;
        ListNode* prev = dummyNode;
        ListNode* cur = head;
        // 找到m位置的节点
        for(int i = 1;i < m;++i){
            prev = cur;
            cur = cur->next;
        }
        // 从m反转到n
        for(int i=m;i<n;++i){
            ListNode* tmp = cur->next;
            cur->next = tmp->next;
            tmp->next = prev->next;
            prev->next = tmp;
        }
        return dummyNode->next;
    }
};
```



#### <3> [BM7-链表中环的入口结点](https://www.nowcoder.com/practice/253d2c59ec3e4bc68da16833f79a38e4?tpId=295&tqId=23449&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

这个题目的哈希表法是我之前没想到的，但是双指针法（快慢指针法）我是了然如胸了！

```c++
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
// way1:双指针法（快慢指针法，很好想的，你其实画个图试几下就出来的了）
// time:O(n),最坏情况下是无环遍历了整个list，space:O（1）,常数空间
class Solution {
public:
    ListNode* EntryNodeOfLoop(ListNode* pHead) {
        // 分2步走：
        // 1-先判断是否有环，无环就马上返回nullptr，有环才继续进行步骤2
        // 2-找环的入口节点指针并返回（双指针法）
        
        // procedure1:
        ListNode* cur1 = pHead,* cur2 = pHead;
        bool hasCircuit = false;// 判断有无环的标志
        while(cur2->next && cur2->next->next){
            cur1 = cur1->next;
            cur2 = cur2->next->next;
            if(cur1 == cur2){
                hasCircuit = true;break;
            }
        }
        if(hasCircuit==false)return nullptr;
        // procedure2:
        ListNode* cur3 = pHead,* cur4 = cur2;
        while(cur3 != cur4){
            cur3 = cur3->next;
            cur4 = cur4->next;
        }
        return cur3;// 此时跳出循环后必然能够找到环入口节点指针！
    }
};

// way2:哈希表法(甚至比双指针法还好想的呢！)
// time:O(n),最坏情况下是无环遍历了整个list，space:O（n）,最坏情况下list中的all节点都存入了uset中
class Solution {
public:
    ListNode* EntryNodeOfLoop(ListNode* pHead) {
        // 思路：将list中all的节点都加入到哈希表中，若当前节点存在于哈希表中则表明存在环，
        // 并且这个第一个存在于哈希表中的节点就是环的入口节点了！
        unordered_set<ListNode*> uset;
        ListNode* cur = pHead;
        while(cur){// 遍历list的同时将每个节点加入到哈希表中
            auto it = uset.find(cur);
            if(it != uset.end())return (*it);
            uset.insert(cur);
            cur = cur->next;
        }
        return nullptr;// 无环，返回空指针节点
    }
};
```



#### <4> [BM11-链表相加(二)](https://www.nowcoder.com/practice/c56f6c70fb3f4849bc56e33ff2a50b6b?tpId=295&tqId=1008772&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

这个题目我第一次do的时候确实没do对！（有暴力的思路也写出来代码了但是通过不了！）

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
// time:O(max(len1,len2))+O(m)+O(n) == O(max(m,n)),反转操作会遍历list1和list2各一次,so
// 取最长的长度即为新的结果list的长度了！
// space:O(1),常量级指针，没有额外的辅助空间
class Solution {
public:
    /**
     * 
     * @param head1 ListNode类 
     * @param head2 ListNode类 
     * @return ListNode类
     */
    ListNode* addInList(ListNode* head1, ListNode* head2) {
        // 看了官方的answer，感觉贼牛逼！
        // 思路：使用了反转list + 加法进位
        
        // 先处理特殊case：
        if(head1==nullptr)return head2;// 一个list为空，相当于0+任何数==任何数！
        if(head2==nullptr)return head1;// 同理
        // 后反转2个list
        head1 = listReverse(head1);
        head2 = listReverse(head2);
        int jinwei = 0;// 进位
        ListNode* newListHead = new ListNode(-1);// 结果list的头节点
        ListNode* cur = newListHead;// 必须要有一个cur节点来遍历！留着这个虚拟头结点do最后的反转操作！
        while(head1 != nullptr || head2 != nullptr){// 2个list中只要有一个不为空就继续do list的加法！
            int val = jinwei;// val记录了每次2个list对应node处->val之和！
            if(head1){
                val += head1->val;
                head1 = head1->next;
            }
            if(head2){
                val += head2->val;
                head2 = head2->next;
            }
            jinwei = val / 10;// 更新进位 的值
            int newVal = val % 10;// push进新的结果list的值！
            cur->next = new ListNode(newVal);
            cur = cur->next;
        }
        if(jinwei != 0){// 只要进位不为0，说明还是有进位值，需要多给list加个新的进位节点，且 节点值为1！
            cur->next = new ListNode(jinwei);
            cur = cur->next;// cur走不走多一步其实都无所谓了的，你画个图即可理解了！
        }
        return  listReverse(newListHead->next);;
    }
    ListNode*  listReverse(ListNode* head){// 子函数，用于反转list，便于从后往前do list->val 的加法！
        if(head==nullptr)return head;
        ListNode* cur = head;
        ListNode* prev = nullptr;
        while(cur){
            ListNode* tmp = cur->next;
            cur->next = prev;
            prev = cur;
            cur = tmp;
        }
        return prev;
    }
};
```



#### <5> [BM13-判断一个链表是否为回文结构](https://www.nowcoder.com/practice/3fed228444e740c8be66232ce8b87c2f?tpId=295&tqId=1008769&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

傻了我，一时间居然没do出来这种简单的题目！

```c++
/**
 * struct ListNode {
 *	int val;
 *	struct ListNode *next;
 * };
 */
// 这个方法超时了！
// class Solution {
// public:
//     /**
//      * 
//      * @param head ListNode类 the head
//      * @return bool布尔型
//      */
//     bool isPail(ListNode* head) {
//         // write code here
//         ListNode* tmp = reverseList(head);
//         ListNode* left = head,* right = tmp;
//         while(left && right){
//             if(left->val != right->val)return false;
//             left = left->next;
//             right = right->next;
//         }
//         return true;
//     }
//     ListNode* reverseList(ListNode* head){
//         ListNode* cur = head,* prev = nullptr;
//         while(cur){
//             ListNode* tmp = cur;
//             cur->next = prev;
//             prev = cur;
//             cur = tmp;
//         }
//         return prev;
//     }
// };

// time:O(n),n是list的长度；space:O(n),n是list的长度；
class Solution {
public:
    bool isPail(ListNode* head) {
        // 思路：将list的元素们转换为数组，用数组是否是回文来判断list是否是回文即可！
        vector<int> v;
        while(head){
            v.push_back(head->val);
            head = head->next;
        }
        return isHuiwen(v);
    }
    bool isHuiwen(const vector<int>& v){
        int l = 0,r = v.size()-1;
        while(l < r){
            if(v[l] != v[r])return false;
            l++,r--;
        }
        return true;
    }
};
```



#### <6> [BM16-删除有序链表中重复的元素-II](https://www.nowcoder.com/practice/71cef9f8b5564579bf7ed93fbe0b2024?tpId=295&tqId=663&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

```c++
/**
 * struct ListNode {
 *	int val;
 *	struct ListNode *next;
 * };
 */
// way1:哈希表法
// time:O(n),n是list的长度;
// space:O(n),哈希表最多需要存储整个list的元素！so哈希表长度为n;
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        // 只要有重复就把这个节点干掉！
        if(head==nullptr)return head;
        map<int,int> umap;
        ListNode* cur = head;
        while(cur){
            umap[cur->val]++; 
            cur = cur->next;
        }
        return buildList(umap);
    }
    ListNode* buildList(const map<int,int>& umap){
        ListNode* dummyNode = new ListNode(-1);
        ListNode* cur = dummyNode;
        for(auto num : umap){
            if(num.second > 1)continue;// 出现次数超过一次就马上跳过此num
            cur->next = new ListNode(num.first);
            cur = cur->next;
        }
        return dummyNode->next;
    }
};
// way2:直接删除法(看的牛客官方的答案，nb！)
// time:O(n),n是list的长度,最坏情况下需要遍历整个list;
// space:O(1),常量级空间;
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        // 直接删除法！
        if(head==nullptr)return head;
        ListNode* dummyNode = new ListNode(-1);
        dummyNode->next = head;
        ListNode* cur = dummyNode;
        while(cur->next && cur->next->next){
            if(cur->next->val == cur->next->next->val){
                int sameValue = cur->next->val;
                while(cur->next && cur->next->val == sameValue){
                    cur->next = cur->next->next;
                }
            }else cur = cur->next;// 继续正常遍历即可!
        }
        return dummyNode->next;
    }
};
```





#### <7> [BM18-二维数组中的查找](https://www.nowcoder.com/practice/abc3fe2ce8e146608e868a70efebf62e?tpId=295&tqId=23256&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

这个题目我在leetcode做过一次，但是忘记怎么做了！

```c++
// 先展示一下错误思路：
class Solution {
public:
    // 我这里自己写的代码就是赌定了某个数字一定会出现在某一行，实则不然！你看看另外的例子就知道了！
    // 比如：7,[[1,2,8,9],[4,7,10,13]],7虽然<9,但是7不在第一个vector中，而在第二个vector中！
    // so我的解题思想是有问题的！
    bool Find(int target, vector<vector<int> > array) {
        // 二维数组的二分法
        // 先找行，再找对应的列，分开来do，先定一边，再定另一边这样子
        int raw = array.size(),col = array[0].size();
        if(raw == 0 || col == 0)return false;
        // 处理特殊case:确保target在这个二维数组的数据范围之内！
        if(target > array[raw-1][col-1])return false;
        if(target < array[0][0])return false;
        int t = -1;
        for(int i = 0;i < raw;++i){
            if(target < array[i][col-1]){
                t = i;break;
            }
        }
        return binarySearch(array[t],target);
    }
    bool binarySearch(const vector<int>& nums,int tar){
        int l = 0,r = nums.size()-1;
        while(l <= r){
            int mid = l+(r-l)/2;
            if(nums[mid]==tar)return true;
            else if(nums[mid] < tar)l = mid+1;
            else r = mid - 1;
        }
        return false;
    }
};
// 正确的做法：（来自牛客官方解答）

// time:O(raw+col),最坏情况下需要遍历二维数组的所有行和所有列；space:O(1),没有用到额外的辅助空间(arr这个二维数组除外)，只占用了常量级内存空间
class Solution {
public:
    bool Find(int target, vector<vector<int> > array) {
        // 二维数组的二分法
        // 先找行，再找对应的列，分开来do，先定一边，再定另一边这样子
        // 利用了二维数组的各个子一维数组是升序数组的特性来deal的，左下角的元素总是比其下方元素小，且比右方元素小
        // 右上角的元素总是比其下方元素小，但比其左方元素大，利用这种特性即可！
        int raw = array.size(),col = array[0].size();
        if(raw == 0 || col == 0)return false;// 处理特殊case
        if(target < array[0][0] || target > array[raw-1][col-1])return false;
        int i = raw-1,j = 0;
        while(i >= 0 && j < col){
            if(array[i][j] == target)return true;
            else if(array[i][j] < target)j++;// 当前元素值 比 目标值 小，往右边走
            else i--;// 当前元素值 比 目标值 大，往上边走
        }
        return false;
    }
};
```



#### <8> [BM19-寻找峰值](https://www.nowcoder.com/practice/fcf87540c4f347bcb4cf720b5b350c76?tpId=295&tqId=2227748&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

这道题目其实本质上 和在数组中查找target目标值没区别！但是我没想到用二分（时间复杂度是O(log2n)==O(logn)）来do！

但凡是见到**==时间复杂度==**要求是**==O(logn)==**的题目，应该第一时间想到用**==二分查找法==**才对！

```c++
// 我自己想的方法：
// time:O(size),一次遍历数组即可完成，space:O(1),常量级空间
class Solution {
public: 
    int findPeakElement(vector<int>& nums) {
        // write code here
        int size = nums.size(),res = -1;
        if(size == 0)return res;
        else if(size == 1)return 0;
        for(int i = 0;i < size;i++){
            // 先处理一下边界case
            if(i == 0){// 是首元素时：只需要判断其是否大于其右边一个元素即可，因为此时nums[-1]=-无穷
                if(nums[i] > nums[i+1]){// 虽说nums[-1]是越界不存在的，但是题目就是这么个意思而已！
                    res = i;break;
                }
                else continue;
            }
            if(i == size-1){// 是尾元素时：只需要判断其是否大于其左边一个元素即可，因为此时nums[size]=-无穷
                if(nums[i-1] < nums[i]){// 虽说nums[size]是越界不存在的，但是题目就是这么个意思而已！
                    res = i;break;
                }
                else continue;
            }
            if(nums[i] > nums[i-1] && nums[i] > nums[i+1]){
                res = i;break;
            }
        }
        return res;
    }
};

// 牛客官网学的二分法：
// time:O(log2n)==O(logn),space:O(1)
// 二分法最坏的情况下是连续对整个数组进行二分，反指数级时间就是log2n，也即分log2n次！
class Solution {
public: 
    int findPeakElement(vector<int>& nums) {
        // 整体思想：只要找到一个峰值满足题目的条件即可，本质上和二分查找tar目标值没什么区别！
        int size = nums.size();
        int l = 0,r = size-1;
        while(l < r){
            int mid = l + (r-l)/2;
            if(nums[mid] > nums[mid+1]){
                // 右边是往下走的，不一定有峰值！so往左边走继续找峰值index！
                r = mid;
            }else{
                // 右边是往上走的，一定存在一个峰值！so往右边走继续找峰值index！
                l = mid + 1;
            }
        }
        return r;// or return l; 均!ok最后左指针l或者右指针r停留的位置均是峰值的index了！(根据样例画个图就出来了)
    }
};
```



#### <9> [BM22-比较版本号](https://www.nowcoder.com/practice/2b317e02f14247a49ffdbdba315459e7?tpId=295&tqId=1024572&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

这个题目主要是巩固了我==**双指针**==的用法！牛客网的题解NB！！！

![image-20220906230847100](C:/Users/11602/AppData/Roaming/Typora/typora-user-images/image-20220906230847100.png)

对应==**双指针**==这种解题思路，只可能有如下几种情况：（你完全排列来想一想也能想出来双指针法怎么个思路）

- 1-两个指针同方向扫描两个链表（或其他数据结构）

- 2-两个指针同方向扫描一个链表

- 2.1-同步走

- 2.2-异步走（即：快慢指针，一个走的快，一个走的慢）

- 3-两个指针反方向扫描（对撞指针）

```c++
// 看了牛客官方的答案才do出来的！(need 按照该算法思路拿个例子来画一遍完整的图例才能理解好！~)
// time:O(max(n1,n2)),最坏的情况下，需要遍历完字符串v1(长度是n1)和v2(长度是n2)才能判断得出结果，
// 那么时间复杂度显然就是两个字符串长度的最大值
// space:O(1),常量级空间，没使用额外的辅助空间
class Solution {
public:
    int compare(string version1, string version2) {
        int n1 = version1.size();
        int n2 = version2.size();
        int i = 0,j = 0;
        // 双指针截取比较法:
        /*
           2个指针同时同向 遍历字符串v1和v2,遍历的同时，
           根据题目的规则 来 判断 每个'.'号之间的数字是否一样 即可得出结论！
        */
        while(i < n1 || j < n2){
            // 注意：是||或号，因为一旦有一个字符串遍历完还得看另一个的数字怎么样
            
            // 先 截取v1的数字用来比较
            // 但是这个已遍历完成的字符串对应的'.'号之间的数字可认为默认是0了！
            long long num1 = 0;// 怕测试用例的整数太大了用int or long不够放(表示不了)！so 用long long
            // 计算v1中 每个.号之间的数字：（用来比大小）
            while(i < n1 && version1[i] != '.'){
                num1 = num1 * 10 + (version1[i] - '0');// 其中，str[i] - '0' == 该字符对于的阿拉伯数字！
                i++;// 往后递增
            }
            // 若遇到逗号了，就跳过这位.号(直接向后++即可)
            i++;
            // 后 截取v2的数字用来比较
            // 计算v2中 每个.号之间的数字：（用来比大小）
            long long num2 = 0;// 怕测试用例的整数太大了用int or long不够放！so 用long long
            while(j < n2 && version2[j] != '.'){
                num2 = num2 * 10 + (version2[j] - '0');
                j++;// 往后递增
            }
            // 若遇到逗号了，就跳过这位.号(直接向后++即可)
            j++;
            // 根据题目的判断规则来 判断 v1和v2的大小关系
            if(num1 < num2)return -1;
            else if(num1 > num2) return 1;
        }
        // 若遍历完都没有返回，则表明v1==v2！return 0即可了！
        return 0;
    }
};
```



#### <10> [BM20-数组中的逆序对](https://www.nowcoder.com/practice/96bd6684e04a44eb80e6a68efc0ec6c5?tpId=295&tqId=23260&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

这个题目我在leetcode上也do过，当时觉得用归并排序来do真的太巧妙了！然后牛客这里的题解也让我更加明白why这样了！好题！

这个方法真是==太巧妙了！==用牛客官方给出的[1 2 3 4 5 6 7 0]来do一次归并的画图操作你就能完全明白why要这么do了！

```c++
// time:O(nlogn),调用n次递归栈空间;
// space:O(n),用了额外的辅助空间临时数组tmp且tmp的长度和原数组一样都是n
class Solution {
private:
    const int mod = 1000000007;
public:
    int InversePairs(vector<int> data) {
        int size = data.size(),res = 0;// res保存原数组总共的逆序对个数！
        /* 思路分析：why要用归并呢？那时间复杂度是O(nlogn)的话应该是要用归并法！
         首先，暴力法是O(n^2)的时间复杂度（2个for），显然不符合题目的要求！
         其次，真正的reason其实是，不论是data == [4 3 1 2] 还是 [3 4 2 1]
                            /     \         /   \
                          [4 3]   [1 2]  [3 4]  [2 1] 这里4与[1 2]构成2个逆序对，那么 3也和 [1 2]构成2个逆序		  对，总共就是4个逆序对了！
         也就是说区间的有序和无序对于逆序对的构成不影响！但有序的话可以方便统计逆序对个数！比如[3 4] [1 2]的话可以		     一次性地把3 和 4 的逆序对个数都统计完！
                          /   \    /  \  /   \   /  \ 
                         4    3  1   2  3    4  2   1
        */ 
        vector<int> tmp(size);
        mergeSort(data,0,size-1,tmp,res);
        return res;
    }
    void mergeSort(vector<int>& data,int l,int r,vector<int>& tmp,int& res){
        // 归并：先递归分割左右子区间，再递归合并左右子区间进行排序！
        // 排序的时候再一并统计逆序对个数即可了！
        if(l >= r){
            return;// 只有一个元素时本身就有序了，无需继续排序!
        }
//         int mid = l + ((r-l) >> 1);
        int mid = l + (r-l)/2;
        // 递归分割左子区间
        mergeSort(data,l,mid,tmp,res);
        // 递归分割右子区间
        mergeSort(data,mid+1,r,tmp,res);
        // 在递归合并各个左右子区间
        merge(data,l,mid,r,tmp,res);
        return;
    }
    void merge(vector<int>& data,int l,int mid,int r,vector<int>& tmp,int& res){
        int lpos = l,rpos = mid+1,pos = l;
        // 进行合并（升序合并 且 统计 逆序对个数）
        while(lpos <= mid && rpos <= r){
            if(data[lpos] >= data[rpos]){
                res += (mid - lpos + 1);// 统计逆序对个数
                res %= mod;// 这是为了防止统计的中途由于逆序对个数太多导致res % mod值改变了
                tmp[pos++] = data[rpos++];
            }else{
                tmp[pos++] = data[lpos++];
            }
        }
        // 合并(if 尚未完成合并的)左子区间到临时数组中！构成有序子区间!
        while(lpos <= mid)tmp[pos++] = data[lpos++];
        // 合并(if 尚未完成合并的)右子区间到临时数组中！构成有序子区间!
        while(rpos <= r)tmp[pos++] = data[rpos++];
        // 将本轮有序子区间覆盖回原来的数组中，以改变原数组无序的现状
        for(int i = l;i <= r;++i)data[i] = tmp[i];
    }
};
```



#### <11> [BM29-二叉树中和为某一值的路径(一)](https://www.nowcoder.com/practice/508378c0823c423baa723ce448cbfd0c?tpId=295&tqId=634&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

忽略了**当前节点是否为空**这一**重要的二叉树递归判断条件**！不应该！

```c++
/**
 * struct TreeNode {
 *	int val;
 *	struct TreeNode *left;
 *	struct TreeNode *right;
 * };
 */
class Solution {
public:
    bool preOrder(TreeNode* cur,int tar){
        if(cur == nullptr)return false;// 若当前节点是空节点时直接返回false（这一点是我第一次写的时候忽略了的！）
        if(cur->left == nullptr && cur->right == nullptr){
            // 若当前节点是叶子节点时 就判断是否符合题目条件
            if(tar - cur->val == 0)return true;
            else return false;
        }
        // 只要找到一条 从根节点 到 叶子节点 sum值==tar的路径就直接返回true即可！
        bool leftHasPath = preOrder(cur->left,tar - cur->val);
        if(leftHasPath)return true;
        bool rightHasPath = preOrder(cur->right,tar - cur->val);
        if(rightHasPath)return true;
        return false;
    }
    bool hasPathSum(TreeNode* root, int sum) {
        // 这个题目在leetcode我也刷过了！
        // 因为题目定义了路径指的是 从 父节点 到叶子节点的过程
        // 也即一定要遍历到叶子节点为止(找不到得到路径和==sum是另一回事，反正就是要找到叶子节点是了)
        // 符合前序遍历的logic！
        if(root==nullptr)return false;
        return preOrder(root,sum);
    }
};
```





#### <12> [BM30-二叉搜索树与双向链表](https://www.nowcoder.com/practice/947f6eb80d944a84850b0538bf0ec3a5?tpId=295&tqId=23253&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

这个题目好像我在leetcode上也do过，但是忘记怎么do了！但是看牛客网官方的answer简直不要太舒服！！！

```c++
// time:O(n),递归遍历二叉树的all的节点的时间复杂度就是n的！
// space:O(n),本来是O(1)的，因为是在原树原地上进行操作，没有申请新空间！但是递归调用栈占用了n的空间！
// 这个牛客官网的answer，挺好的！值得借鉴！
class Solution {
public:
    TreeNode* pre = nullptr;
    TreeNode* head = nullptr;
    TreeNode* inOrder(TreeNode* cur){
        if(cur == nullptr)return cur;
        // 左 中 右 的中序遍历思路!
        inOrder(cur->left);// 左
        if(pre == nullptr){// 先初始化一下list头和要遍历的当前节点的前一个节点！
            pre = cur;
            head = cur;
        }else{
            // do连接为 双向list的操作！
            pre->right = cur;
            cur->left = pre;
            pre = cur;// 更新pre节点为当前节点！以便于下一次递归的连接操作！
        }
        inOrder(cur->right);// 右
        return head;
    }
    TreeNode* Convert(TreeNode* pRootOfTree) {
        // 首先，对于二叉搜索树，其本身的中序遍历就是有序的！so 必须要采用中序遍历的logic来do！
        if(pRootOfTree == nullptr)return pRootOfTree;
        return inOrder(pRootOfTree);
    }
};
```



####　<13> [BM31-对称的二叉树](https://www.nowcoder.com/practice/ff05d44dfdb04e1d83bdbdab320efbcb?tpId=295&tqId=23452&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

这个题目我do过N次了，但是这次还是没有一次过写出来，没有！而是想了一下才写好的！

这个题目要记住的一个key是：是镜像的BT！就必须要判断一下**left** 和 **right** 才能分别判断出来！

```c++
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
// time:O(n),要遍历二叉树的all节点，,n是节点总数！
// space:O(n),因为调用了递归栈空间，且最大要占用n的递归栈的空间
// 也即：最坏情况下二叉树退化为list，那么递归栈深度为n，就要用n的递归栈空间！
class Solution {
public:
    // 采用后序遍历的思路来do
    bool isJingXiang(TreeNode* left,TreeNode* right){
        if(!left && !right)return true;
        else if(left && !right)return false;
        else if(!left && right)return false;
        else if(left->val != right->val)return false;
        // 此时走到这里，就说明左右子数都不为空 且 值一样！
        bool outSideRes = isJingXiang(left->left,right->right);// 再继续判断其外侧子树 看是否为对称的
        bool inSideRes = isJingXiang(left->right,right->left);// 再继续判断其内侧子树 看是否为对称的
        bool res = outSideRes && inSideRes;// 中
        return res;
    }
    bool isSymmetrical(TreeNode* pRoot) {
        if(pRoot == nullptr)return true;// 空树也可以认为是一种特殊的镜像二叉树！
        return isJingXiang(pRoot->left,pRoot->right);
    }
};
```



#### <14> [BM32-合并二叉树](https://www.nowcoder.com/practice/7298353c24cc42e3bd5f0e0bd3d1d759?tpId=295&tqId=1025038&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

这道题目写出来不难，难的是理解它的**时间**和**空间复杂度**的计算！

```c++
// time:O(min(n,m)),要遍历整棵二叉树，其中，n和m分别是两棵树的深度,
// 当一棵树访问完时，自然就连接上另一棵树的节点，故只访问了小树的节点数.
// space:O(min(n,m)),递归栈的深度 也 同时间，只访问了小树的节点数！
class Solution {
public:
    // 这里以树1作为蓝本来合并两棵树！
    // 当然，你用树2作为蓝本来do也是ok的！
    TreeNode* traversal(TreeNode* cur1,TreeNode* cur2){
        if(cur1 == nullptr)return cur2;// 若树1当前节点为空，则返回树2的节点 即可
        if(cur2 == nullptr)return cur1;// 若树2当前节点位空，则返回树1的节点 即可
        // 直接采用前序遍历的logic来do即可了！
        // 中
        cur1->val += cur2->val;
        // 左 右
        cur1->left = traversal(cur1->left,cur2->left);
        cur1->right = traversal(cur1->right,cur2->right);
        return cur1;
    }
    TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2) {
        return traversal(t1,t2);
    }
};
```



#### <15> [BM34-判断是不是二叉搜索树](https://www.nowcoder.com/practice/a69242b39baf45dea217815c7dedb52b?tpId=295&tqId=2288088&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

这道题目写出来不难，难的是理解它的**时间**和**空间复杂度**的计算！

```c++
// time:O(n),n是BST的all节点！最坏情况下需要判断遍历完BT才知道这棵树是否为BST！n是BT的节点总数
// space:O(n),最坏情况下(即BT退化为了List的情况下)调用了深度为n的递归栈空间（n层递归）
class Solution {
public:
    TreeNode* pre = nullptr;// 定义一个当前节点的 前一个节点值，用于do下属BST规则的比较！
    // 对于BST来说，每个节点的节点值都必须要满足: leftValue < curValue < rightValue
    // 并且这个规则 也要适用于 all的BST的子树！
    bool inOrder(TreeNode* cur){
        if(cur == nullptr)return true;
        // 中序遍历的思路来do！即：左 中 右
        bool left = inOrder(cur->left);
        if(pre == nullptr)pre = cur;
        else {
            if(pre->val >= cur->val)return false;// 不是BST了直接就return false！不用继续递归do比较了！
            else pre = cur;// 继续更新pre节点，便于do下一次的递归判断是否为BST！
        }
        bool right = inOrder(cur->right);
        return left && right;
    }
    bool isValidBST(TreeNode* root) {
        // BST的中序遍历是升序的！因此遍历一下取得BST的每个节点值看看是否是否是升序就能够判断是否为BST的！
        if(root == nullptr)return true;// 空树也是特殊的BST!
        return inOrder(root);
    }
};
```



#### <16> [BM35-判断是不是完全二叉树](https://www.nowcoder.com/practice/8daa4dff9e36409abba2adbe413d6fae?tpId=295&tqId=2299105&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

这个题目虽说眼熟，但愣是没有ac出来！！！好好反省自己！！！

```c++
/**
 * struct TreeNode {
 *	int val;
 *	struct TreeNode *left;
 *	struct TreeNode *right;
 *	TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 * };
 */

// space:O(n/2)==O(n),最坏情况下，queue需要存储n/2个节点！
// time:O(n),遍历了二叉树all的节点！其中，n是二叉树节点总数！
class Solution {
public:
    bool isCompleteTree(TreeNode* root) {
        // 思路：对于一棵二叉树，只要存在 其左子树空 但右子树不为空时 就一定不为完全二叉树了！
        if(root == nullptr)return true;// 空树一定是完全二叉树！
        queue<TreeNode*> que;// 用queue来辅助do层序遍历
        que.push(root);
        bool left = true;// 一开始默认左子树不空
        while(!que.empty()){
            int size = que.size();
            for(int i=0;i < size;++i){
                auto t = que.front();
                que.pop();
                if(t == nullptr)left = false;
                else{
                    if(left == false)return left;
                    // 只要发现当前二叉树左子节点空了，但右子节点不空，就说明不是完全二叉树了！
                    // 然后不论3721先把左右子节点都push进去queue队列中去！
                    que.push(t->left);
                    que.push(t->right);
                }          
            }
        }
        return true;
    }
};
```



#### <17> [BM36-判断是不是平衡二叉树](https://www.nowcoder.com/practice/8b3b95850edb4115918ecebdf1b4d222?tpId=295&tqId=23250&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

这个题目我居然能判断错使用前中后序 何种遍历BT的logic！！！该死的！！！

```c++
class Solution {
public:
    int postOrder(TreeNode* cur){
        if(cur == nullptr)return 0;
        //  左 右 中 ，求高度的经典的后序遍历从下到上的logic！
        int leftHeight = postOrder(cur->left);// 左
        if(leftHeight == -1)return -1;
        int rightHeight = postOrder(cur->right);// 右
        if(rightHeight == -1)return -1;
        // 中
        int res = abs(leftHeight - rightHeight) > 1?-1:max(leftHeight,rightHeight)+1;
        return res;
    }
    bool IsBalanced_Solution(TreeNode* pRoot) {
        if(pRoot == nullptr)return true;// 空树也认为是平衡BT！
        // 看到高度差，第一时间想到的是从下到上求高度，也即是后序遍历BT的logic
        return postOrder(pRoot) == -1?false:true;
    }
};
```



#### <18> [BM40-用前序和中序vector重建二叉树](https://www.nowcoder.com/practice/8a19cbe657394eeaac2f6ea9b0f6fcf6?tpId=295&tqId=23282&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

当然，用[后序和中序vector重建BT](https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)也是一样的套路，后续的last一个元素就是当前BT的根节点值！然后也是先分割中序后用中序左Size来分割后序，一样的思路！只是我在刷leetcode这俩题目的时候还记得这个套路，但是刷牛客的题时就不太记得清楚了！需要反复多刷才行！

```c++
/**
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
// s:O(n),最坏情况下需要使用n层递归栈空间
// t:O(n^2),最坏情况下，又需要构成临时的vector，是O(n^2)
class Solution {
public:
    TreeNode* traversal(vector<int> prev,vector<int> in){
        int size = prev.size();
        if(size == 0)return nullptr;
        TreeNode* root = new TreeNode(prev[0]);// 前序的每个首元素都是当前BT的根节点元素！
        if(size == 1)return root;// 剩下一个的话直接build当前root节点即可return了！无需继续递归build BT了！
        int idx = 0;
        while(idx < in.size()){
            if(in[idx] == root->val)break;
            idx++;
        }
        // 整体思路就是先用前序的首元素来分割 中序
        vector<int> leftIn(in.begin(),in.begin()+idx);
        vector<int> rightIn(in.begin()+idx+1,in.end());
        // 物理上直接删除前序首元素！因为已经用过来build当前BT的根节点了！
        prev.erase(prev.begin(),prev.begin()+1);
        // 然后用中序左的size来分割 前序
        int leftInSize = leftIn.size();
        vector<int> leftPrev(prev.begin(),prev.begin()+leftInSize);
        vector<int> rightPrev(prev.begin()+leftInSize,prev.end());
        // 继续递归构造左子树
        root->left = traversal(leftPrev,leftIn);
        // 继续递归构造右子树
        root->right = traversal(rightPrev,rightIn);
        return root;
    }
    TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
        if(pre.size() == 0 || vin.size() == 0)return nullptr;
        return traversal(pre,vin);
    }
};
```





#### <19> [BM43-包含min函数的栈](https://www.nowcoder.com/practice/4c776177d2c04c2494f2555c9fcc1e49?tpId=295&tqId=23268&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)



```c++
// 我自己一开始写的代码：
// time:O(n)
// space:O(n)
class Solution {
private:
    stack<int> stk;
public:
    void push(int value) {
        stk.push(value);
    }
    void pop() {
        stk.pop();
    }
    int top() {
        if(stk.empty())return -1;
        return stk.top();
    }
    int min() {
        stack<int> tmp;
        int res = INT_MAX;
        // 这一个while循环就能求stack的最下值了！
        while(!stk.empty()){
            int t = stk.top();
            stk.pop();
            res = res > t ? t : res;// 保存栈的最小值！
            tmp.push(t);
        }
        // 再把原来stk中的元素push还回去！
        while(!tmp.empty()){
            stk.push(tmp.top());
            tmp.pop();
        }
        return res;
    }
};

// 我自己这个构造出来的min方法时间复杂度是O(n)而不是O(1)了！
// 但是栈的各个操作（push/pop/top都是O(1)的操作），so显然不符合题意！
// 因此需要用 空间 换 时间 的思想了！
// 因为必须要使得栈的操作时间复杂度是O（1），空间复杂度是O（n）
// 因此就必须要采用双栈法！（牛客官方的answer,可在题解区找到官方蓝色字label的answer！）

// time:O(1)
// space:O(n)
class Solution {
private:
    stack<int> stk1,stk2;
    // 其中stk1正常保存push/pop/top操作后的栈元素！
    // stk2保存此时stk1中最下的元素！
public:
    void push(int value) {
        stk1.push(value);
        if( stk2.empty() || value < stk2.top() ){
            stk2.push(value);
        }
        else{
            stk2.push(stk2.top());// 与stk1保持元素个数的一致性！
        }
        // 思路：若stk2为空 或者 当前要插入到元素值 < stk2的栈顶元素值
        // 就把value也加入到stk2中
        // 这样就能保证stk2中的栈顶元素都是stk1中的最小值了！
        // 若 当前要插入到元素值 > stk2的栈顶元素值 时
        // 就把stk2的栈顶元素再插入到stk2中，以保证后面stk1执行了pop操作后stk2与stk1保持一致
        // 这样就不至于说之前stk1的min是-1，但是现在pop掉stk1中的-1后，返回的stk2.top()还是-1
        // 这样子就乱套了的！这是非常容易出错的一个key point！
    }
    void pop() {
        // stk1和stk2都必须要同时同步地pop元素！
        stk1.pop();
        stk2.pop();// 与stk1保持操作的一致性！
    }
    int top() {
        if(stk1.empty())return -1;
        return stk1.top();
    }
    int min() {
        return stk2.top();
    }
};
```





#### <20> [BM44-有效括号序列](https://www.nowcoder.com/practice/37548e94a270412c8b9fb85643c8ccc2?tpId=295&tqId=726&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

这个题目其实在leetCode我是写过好几次的了，但是就是忘记怎么写了！有一点点思路，但是就是忘记了！牛客题解区官方的answer有对应的算法思路动画，你一看就明白的了！！！

```c++
// time:O(n),最坏情况下需要遍历完整个特殊括号字符串s
// space:O(n),用了一个stack的辅助空间来do括号的匹配
class Solution {
public:
    bool isValid(string s) {
        // 原理：最外层的括号 即是 最早出现的左括号 
        // 一定要对应 最晚出现的右括号 也即是先进后出的思路！因此可以用stack这种栈数据结构来deal！
        // 一句话概况思路：最左边先出现的左括号 必须要匹配 最右边晚出现的右括号
        // 否则就是一个无效的括号字符串！
        stack<char> stk;
        for(char ch : s){
            if(ch == '{')stk.push('}');
            else if(ch == '[')stk.push(']');
            else if(ch == '(')stk.push(')');
            else if(stk.empty()){
                // 若左括号没出现的case下就说明右括号先出现了，
                // 那肯定是不符合题目条件的，那就无需继续匹配下去浪费时间了！
                return false;
            }
            else if(ch == stk.top()){// 必须要有左括号的case下才能匹配右括号
                stk.pop();// 符合匹配规则就pop掉当前栈顶元素所代表的右括号即可
            }
        }
        return stk.empty();// if栈为空则是有效的，否则就是无效的！
    }
};
```



#### ==<21>== [BM49-表达式求值](https://www.nowcoder.com/practice/37548e94a270412c8b9fb85643c8ccc2?tpId=295&tqId=726&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)-这个题我目前还是不太会写

这个题目确实非常有难度的，要考虑全面不简单的，比leetCode上面那个“简化版本”的表达式求值难多了！

从这个题目中我可以学到一个==**新的小知识点**==：

```c++
#include<iostream>
#include<cstdio>
int isdigit(char){...}
// 这个isdigit函数可以帮助我们判断一个 字符 是否为 数字字符！
// 是数字字符就返回非0，不是数字字符就返回0
// 例子：
int main(void){
    if(isdigit('1')){
    	cout<<"是数字字符！"<<endl;
    }else{
        cout<<"不是数字字符！"<<endl;
    }
    return 0;
}
// answer: 是数字字符！
```



##### 题目的主要信息：

- 写一个支持+ - \*三种符号的运算器，其中优先级+ - 是一级，\*更高一级
- 支持**括号**运算

**思路：**

对于上述两个要求，我们要考虑的是两点，一是处理运算优先级的问题，二是处理括号的问题。

处理优先级问题，那必定是乘号有着优先运算的权利，加号减号先一边看，我们甚至可以把减号看成加一个数的相反数，则这里只有乘法和加法，那我们优先处理乘法，遇到乘法，把前一个数和后一个数乘起来，遇到加法就把这些数字都暂时存起来，最后乘法处理完了，就剩余加法，把之前存起来的数字都相加就好了。

处理括号的问题，我们可以将括号中的部分看成一个新的表达式，即一个子问题，因此可以将新的表达式递归地求解，得到一个数字，再运算：

- **终止条件：** 每次遇到左括号意味着进入括号子问题进行计算，那么遇到右括号代表这个递归结束。
- **返回值：** 将括号内部的计算结果值返回。
- **本级任务：** 遍历括号里面的字符，进行计算。

```c++
// 我只能说牛逼Plus！！！太tmd难想了吧这个算法思路！我自己脑洞+画图模拟一次才勉强想明白的！
// 太tmd多细节需要考虑了！！！真tmd牛逼！！！
class Solution {
public:
    vector<int> function(string s,int index){
        stack<int> stk;
        int i = -1;// 用于do遍历操作！
        int num = 0;
        char op = '+';// 默认的操作是+号！
        for(i = index;i < s.size();++i){
            // 先将字符数字转换为整形数字
            if(isdigit(s[i])){
                num = num * 10 + (s[i] - '0');
                if( i != s.size() - 1 )continue;// 只要不是遍历的str的末尾了，就继续遍历看是否还有数字！
            }
            // 碰到'('时，把括号内的当成一个数字处理
            if(s[i] == '('){
                // 递归处理括号
                vector<int> res = function(s,i+1);
                num = res[0];// 更新数字num
                i = res[1];// 更新最新遍历的下标i
                if( i != s.size() - 1 )continue;// 只要不是遍历的str的末尾了，就继续遍历看是否还有数字！
            }
            switch(op){
                case '+':{
                    stk.push(num);
                };break;
                case '-':{
                    // push相反数即可！
                    stk.push(-num);
                };break;
                    //优先计算乘号
                case '*':{
                    int tmp = stk.top();
                    stk.pop();
                    stk.push(num * tmp);
                };break;
            }
            num = 0;// 这一步让num 归0 非常 非常 非常 之重要！！！
            // 若 不加这一步的话 上面的 将字符数字转换为整形数字 的操作就会出错！
            // 就没法独立地统计每个数字并变为正确的整形数字了！
            // 右括号结束递归
            if(s[i] == ')'){
                break;
            }else {// 不结束递归就继续更新op操作执行下一步的求值运算！
                op = s[i];
            }
        }
        // 结束求值运算，然后将栈用到元素值依次累加即可！
        int sum = 0;
        while(!stk.empty()){
            sum += stk.top();
            stk.pop();
        }
        return {sum,i};
    }
    int solve(string s) {
        // 求表达式求值！
        return function(s,0)[0];
    }
};
```





#### <22> [BM46-最小的K个数](https://www.nowcoder.com/practice/6a296eb82cf844ca8539b57c23e6e9bf?tpId=295&tqId=23263&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

[优先队列](https://blog.csdn.net/ACM_hades/article/details/89671679)

[c++优先队列(priority_queue)用法详解](https://blog.csdn.net/weixin_36888577/article/details/79937886?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-79937886-blog-89671679.pc_relevant_multi_platform_whitelistv6&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-79937886-blog-89671679.pc_relevant_multi_platform_whitelistv6&utm_relevant_index=2)

这个题目非常不错子的！让我学习到了之前我没有注意到达一个数据结构！优先队列（C++STL里封装好了这种数据结构给我们使用即是：**priority_queue**）

```c++
// time:O(nlogn),最坏情况下，n == k，又因为优先队列增删元素值都是O(logn)的时间复杂度
// 因此总的取前k小元素值的操作的时间复杂度就是O(k*logn) == O(n*logn)
// space:O(n),n是原输入数组的大小
class Solution {
  public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        // 使用优先队列来do事情！
        priority_queue<int, vector<int>, greater<int>> que;
        for (int& num : input)que.push(num);
        vector<int> res;
        while (k--) { // 将最小的四个数字push进结果数组中！
            res.push_back(que.top());
            que.pop();
        }
        return res;
    }
};
```



#### <23> [BM50-两数之和](https://www.nowcoder.com/practice/20ef0972485e41019e39543e8e895b7f?tpId=295&tqId=745&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

这个我之前在leetcode上do过了，但是不太记得解题思路来，so就通过回顾牛客网的思路，一次遍历就能deal这个题目！

```c++
// time:O(n),最坏情况下，需要遍历完整个数组才能刚刚好找到两数之和==target，此时哈希表中存的元素个数就是n-1！
// space:O(n),最坏情况下，哈希表需要存n-1个元素，so O(n-1) == O(n)
class Solution {
  public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        unordered_map<int, int> hashMap;
        for (int idx = 0; idx < numbers.size(); ++idx){
            auto it = hashMap.find(target - numbers[idx]);
            if( it != hashMap.end() ){
                return {it->second+1,idx+1};// 因为能找到就说明前面已经出现过该元素来，那么idx就理应在前面的！
                // 因为题目要求返回的数组下标是从1开始算的，而我们之前都是从0开始算的，那么就需要最后加一！
            }else{
                hashMap[numbers[idx]] = idx;
            }
        }
        return {-1,-1};// 表示没有在数组中找到两数之和 == target的 元素的idx！
    }
};
```



#### <24> [BM50-三数之和](https://www.nowcoder.com/practice/345e2ed5f81d4017bbb8cc6055b0b711?tpId=295&tqId=731&ru=%2Fpractice%2F20ef0972485e41019e39543e8e895b7f&qru=%2Fta%2Fformat-top101%2Fquestion-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%E7%AE%97%E6%B3%95%E7%AF%87%26topicId%3D295)

这个三数之和还得是看carl哥的思路do出来比较容易理解！！！

```c++
class Solution {
public:
    //拿一个 -4 -1 -1 0 0  1 1 2 2 3 3 试一试这段代码！你就可以马上掌握！再不行你也得给老子背下来！这个题目非常常考！
    // 时间复杂度：O(n^2)
    // 空间复杂度：O(n)
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> res;
        int size = nums.size();
        std::sort(nums.begin(),nums.end());
        // 题目目标：找到nums[i]+nums[j]+nums[k] == 0的三元组！
        for(int i = 0;i < size;++i){
            if(nums[i] > 0)break;// 若当前元素都>0了，那么后续元素之和肯定比0大了！那还找什么三元素呢对吧？
            // 给nums[i]去重！
            if(i > 0 && nums[i] == nums[i-1])continue;
            int l = i + 1,r = size - 1;
            while(l < r){
                if(nums[i] + nums[l] + nums[r] < 0)l++;// 太小了，大一点把
                else if(nums[i] + nums[l] + nums[r] > 0)r--;// 太大了，小一点把
                else{
                    res.push_back({nums[i],nums[l],nums[r]});
                    // 然后再对nums[l] 和nums[r] do 去重的操作！
                    while(l < r && nums[l] == nums[l+1])l++;
                    while(l < r && nums[r] == nums[r-1])r--;
                    // 继续让l和r双指针往下一个位置遍历！
                    l++,r--;
                }
            }
        }
        return res;
    }
};
```





#### <25> [四数之和](https://leetcode.cn/problems/4sum/submissions/)



```c++
// 学习carl哥教我的方法！
// 默认升序sort + 双指针法！
// 四数之和，三数之和，手拿把窜！见到一题就给你秒一题！
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        int size = nums.size();
        if(size < 4)return {};
        vector<vector<int>> res;
        std::sort(nums.begin(),nums.end());
        // 本题的目的就是找出 nums[i]+nums[j]+nums[l]+nums[r] == target 的四元组！
        for(int i = 0;i < size;++i){
            // if(nums[i] > target)break;
            // 在这里不能生效！除非target == 0
            // 不然的话这个去重就是错误的！你想想[-4,-2,-1,-1,0,0,4,4,5,5] target == 4
            // 你能够因为中途有一个数字大于4就使得四数之和加起来不能够== target了嘛？天真！
            // 不信就自己去举个例子随便试1试就行，就能知道这句code是怎么错的了！
            if(i > 0 && nums[i] == nums[i-1])continue;// 给 nums[i] 去重！
            for(int j = i + 1;j < size;++j){
                // if(nums[j] > target)break;
                if(j > i + 1 && nums[j] == nums[j-1])continue;// 给 nums[j] 去重！
                int l = j + 1,r = size - 1;
                while(l < r){
                    // 这里 指针 l 必然不能等于 r，因为当前两个指针指向一个位置的元素时，就都代指一个元素了，那么此时我找到的是三元组？傻了吧你！老子找到是四元组！so不能等于！不信你自己画个例子就能理解！
                    
                    // 注意这里，因为测试用例数字累加之后会超过int的范围，so建议使用long保存四数之和的值！
                    if( (long)nums[i]+nums[j]+nums[l]+nums[r] < target )l++;// 太小了，变大一点！
                    else if((long)nums[i]+nums[j]+nums[l]+nums[r] > target)r--;// 太大了，变小一点！
                    else{
                        // 找到其中1个符合条件的四元组了！
                        res.push_back({nums[i],nums[j],nums[l],nums[r]});
                        // 给nums[l] 和 nums[r] 去重！
                        while(l < r && nums[l] == nums[l + 1])l++;
                        while(l < r && nums[r] == nums[r - 1])r--;
                        l++,r--;// 继续让双指针遍历到下一轮的位置！
                    }
                }
            }
        }
        return res;
    }
};
```







#### <26> [BM56-有重复项数字的全排列](https://www.nowcoder.com/practice/a43a2b986ef34843ac4fdd9159b69863?tpId=295&tqId=700&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

这个题目比BM55难一点点，因为BM55是没有重复元素的全排列，不用do同一个树层的去重工作！（我刷这么多次了还是不能一次过AC！）

但是这个题目的**key**就是如何**do同一个树层的去重工作**了！！！（这个你拿个小例子比如:[1,1,2]来画个图差不多能理解去重的代码了！）

```c++
class Solution {
private:
    vector<vector<int>> res;
    vector<int> tmp;
public:
    void backtracking(const vector<int> &nums,vector<int>& used){
        int size = nums.size();
        if(tmp.size() == size){
            res.push_back(tmp);
            return;
        }
        for(int i = 0;i < size;++i){
            // 先do同一树层的去重工作！ 后 do递归+回溯的正常操作！
            if(i>0 && nums[i]==nums[i-1] && used[i-1] == 0)continue;
            // 处理当前节点！
            if(used[i] == 1)continue;// 使用过就直接跳过！不用it来do全排列!
            tmp.push_back(nums[i]);
            used[i] = 1;
            // 继续递归！（当然，有递归就必须要有回溯！）
            backtracking(nums,used);
            // 回溯！
            tmp.pop_back();
            used[i] = 0;
        }
    }
    vector<vector<int> > permuteUnique(vector<int> &num) {
       res.clear();tmp.clear();
       // 因为题目要求了要以字典顺序do升序排列
       // 因此就必须要先do一个升序的sort！
       std::sort(num.begin(),num.end());
       int size = num.size();
       vector<int> used(size);
       backtracking(num,used);
       return res;
    }
};
```





#### <27> [BM57-岛屿数量](https://www.nowcoder.com/practice/0c9664d1554e466aa107d899418e814e?tpId=295&tqId=1024684&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

这个题目说深搜DFS方法确实NB-PLUS的！！！

```c++
// time:O(row*col),最坏的情况下，整个grid原矩阵都是'1'，需要你遍历完整个二维数组才能统计好岛屿的数量！
// space:O(row*col),最坏情况下，整个grid原矩阵都是'1'，需要用到的递归栈深度是row*col，即行*列数，
class Solution {
private:
    // dfs深搜:把(i,j)位置的相邻的上下左右all位置若为'1'的都置为'0'!
    void dfs(vector<vector<char> >& grid,int i,int j){
        int row = grid.size();
        int col = grid[0].size();
        grid[i][j] = '0';
        // 上，下，左，右
        if(i - 1 >= 0  && grid[i-1][j] == '1')dfs(grid,i-1,j);// 深搜 上 do同样的子问题！
        if(i + 1 < row && grid[i+1][j] == '1')dfs(grid,i+1,j);// 深搜 下 do同样的子问题！
        if(j - 1 >= 0  && grid[i][j-1] == '1')dfs(grid,i,j-1);// 深搜 左 do同样的子问题！
        if(j + 1 < col && grid[i][j+1] == '1')dfs(grid,i,j+1);// 深搜 右 do同样的子问题！
    }
    int solve(vector<vector<char> >& grid) {
        int row = grid.size();
        //空矩阵的情况
        if (row == 0)
            return 0;
        int col = grid[0].size();
        //记录岛屿数
        int count = 0;
        //遍历矩阵
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                //遍历到1的情况
                if (grid[i][j] == '1') {
                    //计数
                    count++;
                    //将与这个1相邻的所有1置为0
                    dfs(grid, i, j);
                }
            }
        }
        return count;
    }
};
```







#### ==<28>== [BM60-括号生成](https://leetcode.cn/problems/generate-parentheses/)

这个题目之前leetcode上我自己想出来了，但是在牛客网上面我又想不出来了！！！

我自己都认为自己的方法非常地牛逼！！！

```c++
class Solution {
  public:
    vector<string> res;
    string tmp;
    bool isValidStr(const string& s) {
        // 这是判断一个括号字符串是否为题目要求的合法字符串！
        stack<char> stk;
        if (s[0] == ')')return false;
        stk.push(s[0]);
        for (int i = 1; i < s.length(); ++i) {
            if (s[i] == '(')stk.push(s[i]);
            else if (s[i] == ')') {
                // 当 当前字符数')'时stk辅助栈空间中居然没有'('符合了
                // 那肯定是无效的括号字符串啊！
                if (stk.empty())return false;
                // 若有，则继续匹配，并将stk的'('弹出！继续进行下一轮的匹配！
                else stk.pop();
            }
        }
        return stk.empty();// 用辅助栈空间是否为空来判断是否生成了正确的有效的括号字符串！
    }
    void backtracking(const string& kuohao, int n, vector<int>& used) {
        if (tmp.size() == n * 2) {
            // 收集结果！
            if (isValidStr(tmp))res.push_back(tmp);
            return;
        }
        for (int i = 0; i < kuohao.size(); ++i) {
            // 剪枝
            if (used[i] > n)
                continue; // n对括号，若某个字符个数超过n个那肯定要剪枝！直接continue！
            // 处理当层的节点
            tmp += kuohao[i];
            used[i]++;
            // 继续递归！
            backtracking(kuohao, n, used);
            // 回溯！
            tmp.pop_back();
            used[i]--;
        }
    }
    vector<string> generateParenthesis(int n) {
        res.clear();
        tmp.clear();
        // 就2个符合，统计用'('和')'的次数！用来剪枝！
        string kuohao =
            "()";// 就是从str这个集合中do排列组合 然后得出合法的括号组合！
        vector<int> used(2, 0);
        backtracking(kuohao, n, used);
        return res;
    }
};
```



#### ==<29>== [BM61-矩阵最长递增路径](https://www.nowcoder.com/practice/7a71a88cdf294ce6bdf54c899be967a2?tpId=295&tqId=1076860&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

这个题目我是看的leetCode一个大佬写的优美题解写出来的！这个代码非常地优美！但是真的蛮难想的！且需要考虑**非常全面**才行！

**（需要多次重复才能accept这种记忆化dfs的思路！）**

![image-20220922220136516](C:/Users/11602/AppData/Roaming/Typora/typora-user-images/image-20220922220136516.png)

```c++
// 这是看的评论区的一个大佬的代码才知道要这么写的！
// 这种思路其实是叫做记忆化深度搜索（记忆化dfs）！题解区的这个大佬写的代码 非常地优雅！！！
// time:O(n*m), 最坏情况下 需要遍历整个二维矩阵才能找出这条最长的递增路径！
/*
    比如：
    [1  2  3  4]
    [9  7  6  5]
    [13 12 11 10]
    time:O(3*4)
*/ 
// space:O(n*m),最坏情况下，需要使用n*m层的递归栈空间！
class Solution {
public:
    int longestIncreasingPath(vector<vector<int>>& matrix) {
        int n = matrix.size(),m = matrix[0].size();
        // mark数组是用来 记录已经搜索过的位置，避免重复计算路径！
        vector<vector<int>> mark(n,vector<int>(m,0));
        int res = 0;
        for(int i = 0;i < n;++i){
            for(int j = 0;j < m;++j){
                res = max(res,dfs(matrix,i,j,-1,mark));
            }
        }
        return res;
    }
    int dfs(vector<vector<int>>& matrix,int i,int j,int prev, vector<vector<int>>& mark){
        // 先判断下当前的位置坐标是否越界！
        int n = matrix.size(),m = matrix[0].size();
        if(i < 0 || j < 0 || i >= n || j >= m )return 0;
        // 判断当前位置的数字是否是符合题目说的 是 递增的！
        if(matrix[i][j] <= prev){
            return 0;
        }
         // 记录已经搜索过的位置，避免重复计算
        if (mark[i][j] != 0) {
            return mark[i][j];
        }
        // 记录搜索路径和（上下左右继续do dfs记忆化深搜！）
        mark[i][j] = 1 + max({
            dfs(matrix,i-1,j,matrix[i][j],mark),// 向上 记忆化dfs
            dfs(matrix,i+1,j,matrix[i][j],mark),// 向下 记忆化dfs
            dfs(matrix,i,j-1,matrix[i][j],mark),// 向左 记忆化dfs
            dfs(matrix,i,j+1,matrix[i][j],mark) // 向右 记忆化dfs
        });
        // 上下左右继续深搜dfs！
        return mark[i][j];
    }
};
/*
	注意：
	max(a,b)就只能够求a和b的最大值，但是
	max({a,b,c,d})则可以求a和b和c和d的最大值！
*/
```





#### <30> [BM83-字符串变形](https://www.nowcoder.com/practice/c3120c1c1bc44ad986259c0cf0f0b80e?tpId=295&tqId=44664&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

    // 这个题目的简单版本在leetCode上面可是中等难度的噢！
    // so不简单的哇这个题目！
    // 注意：英文大小写的ASCII码 相差32！
    // 也就是说 'a' == 'A' - 32; 的意思！

```c++
// time:O(2*n)==O(n),n是字符串的长度，因为遍历了两遍整个字符串，so时间复杂度是O(n*2)
// 虽然while里面还嵌套了别的while循环，但是这些内while循环的时间复杂度 相比外while的时间复杂度是非常小的，so可忽略不计！
// space:O(n),辅助空间stack和辅助字符串newS的长度就是n！
// 这是我自己想的，我个人认为比牛客官方answer还好！但是我没有第一次就ac过！还是写了改改了写好几遍才能ac的！
class Solution {
  public:
    string trans(string s, int n) {
        int size = s.length();
        // 因为do 反转操作特别适合 后进先出 的 特点！
        // so使用一个辅助栈结果来do！
        stack<string> stk;
        int i = 0;
        while (i < size) {
            // 找一个一个的word
            // 辅助指针j,用来截取word
            int j = i;
            while (j < size && s[j] != ' ')j++;
            string word = s.substr(i, j - i);
            str_Convert(word);// 先转换了再push进栈中！
            stk.push(word);// 把每个截取出来的字符串放到辅助栈空间中！
            // 去掉中间的多余空格子！以便于
            while (j < size && s[j] == ' '){
                stk.push(" ");// 空格子也是需要收集的！
                j++;
            }
            i = j;// 更新主要用于do遍历的指针i
        }
        // 截取出all的单个word在辅助栈里面之后
        // 就拼接 成 题目要求的新字符串即可！
        string newS = "";
        while (!stk.empty()) {
            string word = stk.top();
            stk.pop();
            newS += word;
        }
        return newS;
    }
    // 大小写 字符 的转换子函数！
    void str_Convert(string& s) {
        int diff_of_DaXiaoXie = 'a' - 'A';// 这个值就是32！
        for (char& ch : s) {
            // 注意：这是char&是为了修改字符串s中的字符！
            if (ch >= 'a' && ch <= 'z') {
                ch = (char)(ch - diff_of_DaXiaoXie);// 小写字符转大写！
            } else if (ch >= 'A' && ch <= 'Z') {
                ch = (char)(ch + diff_of_DaXiaoXie);// 大写字符转小写！
            }
        }
    }
};
```





#### <31> [BM84-最长公共前缀](https://www.nowcoder.com/practice/28eb3175488f4434a4a6207f6f484f47?tpId=295&tqId=732&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

我只能说牛客官方的这个题目的answer着实牛逼Plus的！！！非常容易理解，还画动态图出来让我容易理解！！！very good的！

若不看官方题解的话我肯定想不出来的！！！**实在不行跟着题解的代码和动画画一遍你也就非常理解这个题目的了！**

```c++
// time:O(n*len),其中，n是字符串数组的个数，len指的是公共前缀的长度！也即是最短的字符串长度！
// 因为作为公共的前缀字符串，不可能比字符串数组中的其他任何一个字符串还要长！最多就只是相等而已！
// space:O(1),没有使用额外的空间！
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        int n = strs.size();
        // 处理特殊case：空字符串数组
        if(n == 0 )return "";
        // 因为最长公共前缀的长度不会超过任何一个字符串的长度，因此我们逐位就以第一个字符串为标杆，遍历第一个字符串的所有位置，取出字符
        // 这句话是整个官方题解的key点！！！
        for(int i = 0;i < strs[0].size();++i){
            char tmp = strs[0][i];
            for(int j = 1;j < n;++j){
                // why这里还需要有 i >= strs[j].size()这个判断条件呢？
                // answer:因为作为公共的前缀字符串，不可能比字符串数组中的其他任何一个字符串还要长！最多就只是相等而已！
                // 这个道理要是不太懂的话，直接拿官方的例子1来画个图就明白的了！
                if(i >= strs[j].size() || strs[j][i] != tmp){
                    return strs[0].substr(0,i);
                    // 返回标杆（字符串数组的第一个字符串）字符串的，以0为开头，的共i个字符的字符串作为最长的公共前缀！
                    // 千万要注意：这里返回的是标杆的前缀作为公共前缀！因为就是拿标杆来do对比的！
                }
            }
        }
        return strs[0];// 最长的公共前缀就是该字符串数组的标杆，即是：字符串数组里面的第一个字符串！
    }
};
```



#### <32> [BM86-大数加法](https://www.nowcoder.com/practice/11ae12e8c6fe48f883cad618c2e81475?tpId=295&tqId=1061819&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)



```c++
// 这个题解是我按着题解区一个大佬的java题解然后 不申请额外栈空间写出来的！
// time:O(max(sSize,tSize))
// space:O(1),没有使用额外的空间

class Solution {
  public:
    string solve(string s, string t) {
        int sSize = s.size(), tSize = t.size();
        if (sSize == 0)return t;
        if (tSize == 0)return s;
        // 让2个数字字符串的长度保持一致！方便后续do的相加操作！
        // 然后把新计算出来的数字直接转换为对应的数字字符放到字符串s中，作为结果集即可！
        while (sSize < tSize) {
            s = "0" + s;
            sSize++;
        }
        while (tSize < sSize) {
            t = "0" + t;
            tSize++;
        }
        int jinwei = 0;
        for (int i = sSize - 1, j = tSize - 1; i >= 0 && j >= 0; i--, j--) {
            int tmp = s[i] - '0' + t[j] - '0' + jinwei;// 求本轮和！
            jinwei = tmp / 10;
            int yushu = tmp % 10;
            s[i] = yushu + '0';// 数字转换为对应的数字字符！
        }
        if(jinwei != 0)s = "1" + s;
        return s;
    }
};
```







#### <33> [BM85-验证IP地址](https://www.nowcoder.com/practice/55fb3c68d08d46119f76ae2df7566880?tpId=295&tqId=1024725&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

这个题目贼jb难，别想了吧我觉得！面试的时候do不出来的基本上，很难想！！！

但是这个题目也让我学习到了一个**技巧**：**split字符串分割函数**的实现！(python中有直接封装好的标准库.split()方法)

```c++
//将字符串从符号spliter中分割开，形成字符串数组！
// test: input "102.123.123.456" --> output {"102","123","123","456"}
vector<string> split(string s,string spliter){
    // s 是待分割de字符串，spliter是待用以分割的符号
    vector<string> res;
    // 遍历字符串查找spliter 并 分割数字字符串到res结果数组中
    int i = 0;
    while (i != s.npos) {
        i = s.find(spliter);
        res.push_back(s.substr(0,i));
        s = s.substr(i + 1);// 把已经分割好的字符串物理删除掉！
    }
    return res;
}
```



#### <34> [BM86-合并两个有序的数组](https://www.nowcoder.com/practice/89865d4375634fc484f3a24b7fe65665?tpId=295&tqId=658&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

这个题目我在leetcode上刷过好几次，但是现在这里又忘记思路了！！可恶的！！！不应该这样的呀！！！

```c++
// time:O(n+m),m是A数组长度，n是B数组长度
// space:O(1),没使用额外空间
class Solution {
public:
        // 思路：
        // 从后往前 将A和B中较大者填入到A的末尾，末尾--，继续循环遍历这样子！
        // 若不懂的话随便画个图就能get到的了！
        // 注意这里的越界条件的判断！
        // idxB小于0时 or A数组自己的元素之比较大时 就让A数组自己的元素填到合适的位置
        // 否则 就让A数组的对应位置填充B数组的元素！
        // 注意idxA和idxB是否越界的问题！
    void merge(int A[], int m, int B[], int n) {
        int idxA = m-1,idxB = n-1,pos = m+n-1;
        for(int pos = m + n - 1;pos >= 0;pos--){
            if(idxB < 0 || (idxA >= 0 && A[idxA] >= B[idxB])){
                A[pos] = A[idxA--];
            }else{
                A[pos] = B[idxB--];
            }
        }
        return;
    }
};
```



#### <35> [BM92-最长无重复子数组](https://www.nowcoder.com/practice/b56799ebfd684fb394bd315e89324fb4?tpId=295&tqId=1008889&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

这个题目是看的牛客网官方answer才知道要这么干才能do出来！

```c++
// time:O(n),n是arr长度，最坏情况下需要遍历整个arr一次
// space：O(n),最坏情况下哈希表需要存储n个pair！
// 解题思路：
// 双指针 + hashmap的方法 维护一个滑动窗口，在双指针遍历的同时，顺便记录 最长无重复元素子数组 的长度！
class Solution {
public:
    int maxLength(vector<int>& arr) {
        // 哈希表:用来记录窗口内 非重复的数字
        unordered_map<int,int> mp;
        int res = 0;
        // 设置窗口左右边界left和right
        // right指针优先移动，然后left左指针用来调整滑动窗口的左边界！（if需要的话）
        for(int left = 0,right = 0;right < arr.size();right++){
            // 窗口右移进入哈希表统计出现次数
            mp[arr[right]]++;
            // 当出现次数大于1时，则窗口内有重复
            while(mp[arr[right]] > 1){
                // 窗口左移，同时减去该数字的出现次数
                mp[arr[left++]]--;
            }
            // 维护子数组长度最大值
            res = max(res,right-left+1);// 统计滑动窗口长度！也就是求当前数组最长无重复子数组长度了！
        }
        return res;
    }
};
```



#### <36> [BM94-接雨水](https://www.nowcoder.com/practice/31c1aed01b394f0b8b7734de0324e00f?tpId=295&tqId=1002045&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

这个题目我都不知道写多少遍了，但是就是不太会！！！这种面试高频题目，但是又比较有难度且思维上不好想的题目需要面试前大量刷大量记忆！！！

```c++
// dp法：学carl哥教我的思路！
// time:O(n*3)==O(n),n是原数组长度,使用了3次for循环，每次都是遍历n个长度的数组
// time:O(n*2)==O(n),使用了2个辅助数组空间存放 每个列 的左边最大高度 和 右边最大高度
class Solution {
public:
    long long maxWater(vector<int>& arr) {
        int size = arr.size();
        vector<int> maxLeft(size,0),maxRight(size,0);
        int sum = 0;// 保存雨水总量！
        // 先求每个列左边最大高度(值得注意的是：第一个列左边最大高度是其本身)
        maxLeft[0] = arr[0];
        for(int i = 1;i < size;++i){
            maxLeft[i] = max(maxLeft[i-1],arr[i]);
        }
        // 再求每个列右边最大高度(值得注意的是：最后一个列右边最大高度是其本身)
        maxRight[size-1] = arr[size-1];
        for(int j = size-2;j >=0;--j){
            maxRight[j] = max(maxRight[j+1],arr[j]);
        }
        // 求雨水总和
        for(int i = 0;i < size;++i){
            int tmp = min(maxLeft[i],maxRight[i]) - arr[i];
            if(tmp > 0)sum += tmp;// 雨水体积大于0时才统计倒结果中！
        }
        return sum;
    }
};
```



#### <37> [BM93-盛水最多的容器](https://www.nowcoder.com/practice/3d8d6a8e516e4633a2244d2934e5aa47?tpId=295&tqId=2284579&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

这个题目我明明在之前leetcode上一次过刷过了，但是这里又忘记怎么用双指针来降低时间复杂度了！！！

```c++
// time:O(n),最坏情况下需要遍历整个数组！
// space:O(1),没有使用额外的辅助空间！
class Solution {
public:
    int maxArea(vector<int>& height) {
        // 暴力法肯定就是2个for了！但是超时了！应该尝试使用双指针法来降低时间复杂度！
        // int size = height.size();
        // int res = 0;
        // for(int i = 0;i < size;++i){
        //     for(int j = i+1;j < size;++j){
        //         int tmp = min(height[i],height[j])*(j-i);
        //         res = max(res,tmp);
        //     }
        // }
        // return res;
        
        // 贪心 + 双指针思想（主要还是双指针）！
        int size = height.size();
        int res = 0;
        int left = 0,right = size - 1;
        while(left < right){
            int tmp = min(height[left],height[right]) * (right-left);
            res = max(res,tmp);// 保存最多可能盛放的雨水量！
            // 优先放弃短边，以便于贪得最多可盛放雨水量！
            if(height[left] < height[right])left++;
            else right--;
        }
        return res;
    }
};
```



#### ==<38>== [BM90-最小覆盖子串](https://www.nowcoder.com/practice/c466d480d20c4c7c9d322d12ca7955ac?tpId=295&tqId=670&ru=%2Fpractice%2F3d8d6a8e516e4633a2244d2934e5aa47&qru=%2Fta%2Fformat-top101%2Fquestion-ranking&sourceUrl=%2Fexam%2Foj)

这种其实就是**简单的hard题目**，其实还是能够deal的！把思路想清楚后再记忆一下还是能够do的！

==还是蛮难想的这个思路！想到了还是不太容易写出bug free的codes的！！！==

滑动窗口（本质上就是用unordered_map + 双指针来do）

这是我看的leetcode的视频和题解后，自己按照这种思路并结合题解写出来的！！！

```c++
class Solution {
public:
    // 滑动窗口+哈希表 法！
    unordered_map<char,int> smap;// key-char存放s中的字符，value-int存放该字符对应的出现频次
    unordered_map<char,int> tmap;// key-char存放t中的字符，value-int存放该字符对应的出现频次
    string minWindow(string s, string t) {
        if(s.size() < t.size()){
            return "";// 返回空串
        }
        for(char ch : t)tmap[ch]++;// 统计t串中各个字符出现次数
        int minLen = INT_MAX;
        int start = 0,end = 0,ansStart = -1;// 双指针维护一个滑动窗口！
        for(;end < s.size();end++){// 右指针end 继续下一轮的右移动遍历！
            if(tmap.find(s[end]) != tmap.end()){
                // 或者直接用if( tmap.count(s[end]) ) 来判断也是ok的！
                // if 有的话则count方法会返回非0(相当于true)，无就返回0(相当于false)
                smap[s[end]]++;
            }
            while(isValidWindow() && start <= end){// start <= end 是用来保证start左指针没越界的 条件！
                // 先把有效的滑动窗口（即是有效的覆盖子串长度 更新一下,以便于后续用.substr()方法截取结果子串！）
                if(minLen > end - start + 1){// 这里采用[start,end]左闭右闭区间来do事情！
                    minLen = end - start + 1;
                    ansStart = start;// 保存最终结果的起始位置,以便于后续用.substr()方法截取结果子串！
                }
                // 找到一个有效窗口的case下就要尝试让start--看是否能够缩小滑动窗口
                if(tmap.count(s[start])){
                    smap[s[start]]--;// 左指针右移动，且让其对应s串中的字符出现次数-1
                    // 左指针start 继续下一轮的右移动遍历！
                }
                start++;
            }
        }
        return ansStart == -1 ? "" : s.substr(ansStart,minLen);
    }
    bool isValidWindow(){
        for(auto p : tmap){// 这里必须使用遍历tmap的方法来拿正确判断是否当前滑动窗口统计倒数字都是有效的！
            if(smap[p.first] < p.second)return false;
            // 此时S中的滑动窗口(所代表的覆盖子串)并没有覆盖t中all的字符！so是无效的滑动窗口（也即是无效的覆盖子串）
        }
        return true;
    }
};
```



#### <39> [BM96-主持人调度（二）](https://www.nowcoder.com/practice/4edf6e6d01554870a12f218c94e8a299?tpId=295&tqId=1267319&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

这个题目我一开始题目都没咋看懂！！！

```c++
这个题目绝逼是出错了！！！甚至题解的codes带入自测例子都不对的！
就比如这个例子：2,[[1,2],[3,4]] 这个按照题意很明显只是需要1个主持人就ok了
但是下面这个官方代码却计算出来res == 2！！！这就是问题了！
```



#### <40> [BM95-分糖果问题](https://www.nowcoder.com/practice/76039109dd0b47e994c08d8319faa352?tpId=295&tqId=1008104&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

我只能说牛客官方题解NB-PLUS!!!题解是非常好理解的！！！

```c++
// time:O(n*2)==O(n),n是原数组arr的长度
// space:O(n),使用长度为n的辅助数组空间vector<int> 存放给每个孩子所分配的糖果数量
class Solution {
public:
    int candy(vector<int>& arr) {
        // 我看牛客官方的题解answer步骤，太clear了！NB-PLUS
        int size = arr.size();
        // 先来个辅助数组，保存各个孩子分配的糖果数（默认情况下每个孩子都来一个糖果,这是 题目所要求的）
        vector<int> tmp(size,1);
        // 先从左到右遍历（比较相邻2个孩子的得分）
        // 若右边孩子比左边孩子得分高，则右边孩子糖果数+1
        for(int i = 1; i < size;++i){
            if(arr[i-1] < arr[i]){
                tmp[i] = tmp[i-1]+1;
            }
        }
        // 后从右到左遍历（比较相邻2个孩子的得分）
        int res = tmp[size-1];// 保存最终糖果总数
        for(int i = size-2; i >= 0;--i){
            if(arr[i] > arr[i+1]){
                if(tmp[i] <= tmp[i+1]){
                    tmp[i] = tmp[i+1]+1;
                }
            }
            res += tmp[i];// 在第2次遍历数组时 就顺便统计糖果总数 可以减少一轮for遍历！
        }
        // 若左边孩子比右边孩子得分高，则左边孩子糖果数+1(if左边孩子糖果数不大于右边孩子糖果数的case下)
        return res;
    }
};
```



#### <41> [BM98-螺旋矩阵](https://www.nowcoder.com/practice/7edf70f2d29c4b599693dc3aaeea1d31?tpId=295&tqId=693&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

这个题我在leetcode上面是do过的了！但还是记不得怎么模拟！！！

```c++
// time:O(m*n),相当于要遍历整个矩阵的all元素了！
// space:O(1),res是必要的结果区间，但除此之外我们并没有使用额外的非必要的辅助空间！
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int> > &matrix) {
        int row = matrix.size();
        if(row == 0)return {};
        int col = matrix[0].size();
        vector<int> res;
        int left = 0,right = col - 1;
        int up = 0,down = row - 1;
        while(left <= right && up <= down){
            // 从左到右
            for(int i = left;i <= right;++i){
                res.push_back(matrix[up][i]);
            }
            up++;
            if(up > down)break;
            // 从上到下
            for(int i = up;i <= down;++i){
                res.push_back(matrix[i][right]);
            }
            right--;
            if(right < left)break;
            // 从右到左
            for(int i = right;i >= left;--i){
                res.push_back(matrix[down][i]);
            }
            down--;
            if(down < up)break;
            // 从下到上
            for(int i = down;i >= up;i--){
                res.push_back(matrix[i][left]);
            }
            left++;
            if(left > right)break;
        }
        return res;
    }
};
```







#### <42> [BM97-旋转数组](https://www.nowcoder.com/practice/e19927a8fd5d477794dac67096862042?tpId=295&tqId=1024689&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

这个我也在leetcode上面已经do过了！但是就是忘记思路是什么了！

```c++
// time:O(3*n)==O(n),3次std::reverse()反转操作 最坏的时间复杂度 都是O(n)
// space:O(1)，并没有使用额外的辅助空间！
class Solution {
public:
    vector<int> solve(int n, int m, vector<int>& a) {
        // 将数组后m个数字覆盖到数组开头处
        // 然后将开头
        if(m == 0)return a;
        m %= n;
        // 因为m有可能会大于n，但是循环右移n次后相当于和原数组是一样的，因此就do一次取余数的操作，过滤掉不必要的右移位的操作！
        std::reverse(a.begin(), a.end());
        std::reverse(a.begin(),a.begin()+m);
        std::reverse(a.begin()+m,a.end());
        return a;
    }
};
```



#### <43> [BM99-顺时针旋转数组](https://www.nowcoder.com/practice/2e95333fbdd4451395066957e24909cc?tpId=295&tqId=25283&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

这个题目==我自己想出来了==！！！你自己画个图也能想出来的！不难！但是就是第一次do这种模拟题目，比较陌生，so应该记录在刷题笔记中，以备忘记这种思路！！！

```c++
// time:O(n*n),相当于遍历了整个矩阵
// space:O(1),除了必要的结果数组外，没有使用额外的辅助空间！
class Solution {
public:
    vector<vector<int> > rotateMatrix(vector<vector<int> > mat, int n) {
        // easy job!
        // 1:先交换对应行
        int l = 0,r = n-1;
        while(l < r){
            swap(mat[l],mat[r]);
            l++,r--;
        }
        // 2:然后swap对角线两边的元素即可了！
        for(int i = 0;i < n;++i){
            for(int j = i+1;j < n;++j){
                // 对角线上到元素无需交换
                // 只是把对角线两边的元素交换即可！
                swap(mat[i][j],mat[j][i]);
            }
        }
        return mat;
    }
};
```



#### <44> [BM64- 最小花费爬楼梯](https://www.nowcoder.com/practice/6fe0302a058a4e4a834ee44af88435c7?tpId=295&tqId=2366451&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

这个题目carl哥总结的dp里面也讲解过，但是！这里牛客网的答案==显而易见==且==非常容易理解==！！！

so我甚至认为牛客网的dp题解比carl哥的题解要好！！！

```c++
// time:O(n),其中n是cost数组的大小,一个for遍历了一轮cost数组，so时间复杂度是O(n)!
// space:O(n),使用了辅助数组空间 vector<int>
class Solution {
public:
    int minCostClimbingStairs(vector<int>& cost) {
        int size = cost.size();
        vector<int> dp(size+1,0);
        // 其中，dp[i]:表示的是到达第i个台阶需要花费的最小代价！
        dp[0] = 0,dp[1] = 0;// 由于第0个 or 第1个台阶无需爬都可以到达，因此这2个台阶的最小代价就是0了！
        for(int i = 2;i <= size;++i){
            dp[i] = min(dp[i-1] + cost[i-1],dp[i-2]+cost[i-2]);// 递归公式！
            // 第i个台阶的最小花费是 第i-1个台阶的最小花费 或者 第i-2个台阶的最小花费！
        }
        return dp[size];
    }
};
```



#### <45> [BM66- 最长公共子串](https://www.nowcoder.com/practice/f33f5adc55f444baa0e0ca87ad8a6aac?tpId=295&tqId=991150&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

这个题目要用dp来do确实很有难度，但是牛客的官方题解还算能理解的，主要 需要我画个图理解下这个代码的思路才能记住这种套路！！！

![2](C:/Users/11602/Desktop/LeetCodeOfferNotes/git/%E7%AE%97%E6%B3%95%E5%AD%A6%E4%B9%A0%E6%88%AA%E5%9B%BE/2.jpg)

```c++
// time:O(len1*len2),遍历对比2个字符串中的每一个字符！
// space:O(len1*len2),辅助dp数组大小是len1*len2的,其中len1是str1的长度，len2是str2的长度
class Solution {
public:
    // 我只能说，这个题解NB-PLUS!太牛逼了！我画个图才能明白这个代码的思路！！！
    string LCS(string str1, string str2) {
        //dp[i][j]表示到str1第i个个到str2第j个为止的公共子串长度
        vector<vector<int> > dp(str1.length() + 1, vector<int>(str2.length() + 1, 0)); 
        int max = 0;
        int pos = 0;
        for(int i = 1; i <= str1.length(); i++){
            for(int j = 1; j <= str2.length(); j++){
                //如果该两位相同
                if(str1[i - 1] == str2[j - 1]){ 
                    //则增加长度
                    dp[i][j] = dp[i - 1][j - 1] + 1; 
                }
                else{ 
                    //该位置为0
                    dp[i][j] = 0; 
                }
                //更新最大长度
                if(dp[i][j] > max){ 
                    max = dp[i][j];
                    pos = i - 1;
                }
            }
        }
        return str1.substr(pos - max + 1, max);
    }
};
// 我自己理解了上面牛客官方给出的代码后写出来带注释的版本：
class Solution {
public:
    // 我只能说，这个题解NB-PLUS!太牛逼了！我画个图才能明白这个代码的思路！！！
    string LCS(string str1, string str2) {
        //dp[i][j]:表示的是str1的到idx==i为止的子串与str2中到idx==j为止的子串中，最长的公共子串长度
        vector<vector<int>> dp(str1.size()+1,vector<int>(str2.size()+1,0));
        int max = 0;// 保存最长公共子串的长度！
        int pos = 0;// 保存str1中是最长公共子串的i的位置！
        // 这里以str1为基准，最终会返回str1中的子字符串作为结果string！
        // so 用pos变量记录str1中的所统计到达最长公共子串的i的位置！
        for(int i = 1;i <= str1.size();++i){
            for(int j = 1;j <= str2.size();++j){
                if(str1[i-1] == str2[j-1]){
                    // 当前字符相等，则最长公共子串长度+1！dp相应位置置为其左上角的dp[i-1][j-1]+1!
                    dp[i][j] = dp[i-1][j-1]+1;
                }else{// 当前字符不相当，dp相应位置置为0
                    dp[i][j] = 0;// 字符不等，此时以i为结尾的str1中的子串与以j结尾的str2中的子串的最长公共子串的长度就为0！
                }
                // 记录str1中的所统计到达最长公共子串的i的位置！
                if(max < dp[i][j]){
                    max = dp[i][j];
                    pos = i - 1;
                }
            }
        }
        return str1.substr(pos - max + 1,max);
    }
};
```



#### <46> [BM70-兑换零钱（一）](https://www.nowcoder.com/practice/3911a20b3f8743058214ceaa099eeb45?tpId=295&tqId=988994&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

我只能说这个思路NB-PLUS!我完全不可能自己想出来！！！你代[5 2 3],20 这个小例子进来这个代码就能理解why是这样写的了！

但要想出来并背下来对我来说还是有挺大难度的！！！

```c++
// time:O(n*aim),n是原货币面值数组长度，aim是目标值！外for从1遍历到aim元，内for遍历整个原货币数组!
// space:O(aim),辅助dp数组大小是aim,so O(aim+1)==O(aim)
class Solution {
public:
    int minMoney(vector<int>& arr, int aim) {
        // 小于1的都返回0
        if(aim < 1)return 0;
        // dp[i]:表示要凑齐i元最少需要多少张货币
        vector<int> dp(aim+1,aim+1);
        dp[0] = 0;// 首先，凑成0元所需最小货币数肯定是0张货币！这个千万不能忘记！
        // 遍历1-aim元
        for(int i = 1;i <= aim;++i){
            // 每种面值的货币都要枚举
            for(int j = 0;j < arr.size();++j){
                // if面值不超过要凑的钱才能用
                if(arr[j] <= i){// 要是大于当前面值为i元的货币你都没法用啊！！！
                    // 维护最小值
                    dp[i] = min(dp[i],dp[i-arr[j]]+1);
                }
            }
        }
        // if最终答案大于aim代表无解(因为这里我们让dp数组都初始化为了aim+1，可以用这个来判断什么时候是无解的！！！)
        return dp[aim] > aim?-1:dp[aim];
    }
};
// 下面我自己根据自己的理解重写一次版本的代码：
// time:O(n*aim),n是原货币面值数组长度，aim是目标值！外for从1遍历到aim元，内for遍历整个原货币数组!
// space:O(aim),辅助dp数组大小是aim+1,so O(aim+1)==O(aim)
class Solution {
public:
    // dp动态规划的本质就是把大的结果问题都分割为若干相互联系的子问题，先deal前面首次出现过的，子问题
    // 然后从这些子问题中得到原问题的结果！
    int minMoney(vector<int>& arr, int aim) {
        // dp[i]:代表要凑成i元所需要最少的货币数
        vector<int> dp(aim+1,aim+1);
        // 初始化dp数组的数值为aim+1是为了方便判断是否能够组成aim
        dp[0] = 0;// 凑成0元所要货币数最少那肯定就是0个货币了！这个千万不能够忘记！
        // 外for遍历1元 ~ aim元，进而通过内for凑成他们所需最少货币数
        for(int i = 1;i <= aim;++i){
            for(int j = 0;j < arr.size();++j){
                // 内for是真正求解当前dp[i]的值的！
                if(arr[j] <= i){
                // 要是大于当前面值为i元的货币你都没法用啊！！！
                // 当前货币面值小于我要凑成的总额时，这张货币 才能够被用上！
                // 这也是保证了 i - arr[j] >= 0 不会使得访问dp数组越界下标出现的条件！
                // 你拿了当前面值为arr[j]的货币来凑i
                // 那肯定使用货币数量会+1啦！！！然后和自身本来的数量对比求最小值即可！
                    dp[i] = min(dp[i-arr[j]]+1,dp[i]);
                }
            }
        }
        // 当dp[aim] > aim时，即是aim+1 > aim这种case发生时，就会判定是无法凑成aim值！
        // so返回-1！（这是题目规定的）
        return dp[aim] > aim ? -1 : dp[aim];
    }
};
```





#### <47> [BM68-矩阵的最小路径和](https://www.nowcoder.com/practice/7d21b6be4c6b429bb92d219341c4f8bb?tpId=295&tqId=1009012&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

这个题目我好像在leetcode上面已经搞过一次了，so这里我很轻松就自己想出来了！和牛客网的官方标准答案一毛一样！！！NB-PLUS!

```c++
// time:O(n*m+n+m),其中,n是matrix矩阵的行数，m是数组的列数
// space:O(n*m),使用了辅助空间dp[i][j],大小是n*m
class Solution {
public:
    // 这道题目其实想一想，把大问题分解为几个相互有练习的子问题
    // 的话，其实也就那样！不难！
    int minPathSum(vector<vector<int> >& matrix) {
        int row = matrix.size(),col = matrix[0].size();
        // dp[i][j]:表示走到(i,j)这个位置的all路径的最小路径和！
        vector<vector<int>> dp(row,vector<int>(col,0));
        // 给dp数组初始化！
        dp[0][0] = matrix[0][0];
        for(int i = 1;i < col;++i){
            dp[0][i] = dp[0][i-1]+matrix[0][i];
        }
        for(int j = 1;j < row;++j){
            dp[j][0] = dp[j-1][0]+matrix[j][0];
        }
        // 正式从(0,0)遍历到(n-1,m-1) 填充dp数组的各个子问题了！
        for(int i = 1;i < row;++i){
            for(int j = 1;j < col;++j){
                // 因为题目规定了只能向右 or 向下走！so取这2个方向路径和的最小值 + 上到达当前(i,j)位置的数字 的总和求生当dp[i][j]的值了！
                dp[i][j] = min(dp[i][j-1],dp[i-1][j])+matrix[i][j];
            }
        }
        return dp[row-1][col-1];// row == n,col == m
    }
};
```



#### <48> [BM72-连续子数组的最大和](https://www.nowcoder.com/practice/459bd355da1549fa8a49e350bf3df484?tpId=295&tqId=23259&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

这种简单dp题目我也没有思路！！！我不能接受这么菜的自己！！！

```c++
// time:O(n),n是原数组长度
// space:O(n),使用了辅助dp空间且其长度为n！
class Solution {
public:
    int FindGreatestSumOfSubArray(vector<int> array) {
        int size = array.size();
        // 先处理特殊case
        if(size == 0)return 0;
        else if(size == 1)return array[0];
        vector<int> dp(size,0);
        // dp[i]:表示终点下标为i的连续子数组的和的最大值就是dp[i]了！
        dp[0] = array[0];// 第一个子数组最大值那肯定就是其本身了！
        int maxSum = INT_MIN;
        for(int i = 1;i < size;++i){
            // 因为要 求的是 连续子数组的和的 最大值
            // so要么当前元素加进去连续子数组和中，只有2种情况需要分析：
            // 1:要么和 变大了
            // 2:要么和 变小了
            // 要是变大了就取dp[i-1]+array[i]作为当前dp[i]的连续子数组和的最大值
            // 要是变小了就截断当前的连续子数组，重新开始求子数组和的最大值！顾名思义就是取array[i]作为当前的以i为止的连续子数组的最大和==dp[i]了！
            dp[i] = max(dp[i-1]+array[i],array[i]);// 求和大就统计起来，否则就重新统计当前值为以当前i下标为止的连续子数组的最大和！
            maxSum = max(maxSum,dp[i]);// 统计结果最大和！
        }
        return maxSum;
    }
};
```



#### <49> [BM71-最长上升子序列(一) ](https://www.nowcoder.com/practice/5164f38b67f846fb8699e9352695cd2f?tpId=295&tqId=2281434&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

真的了，这个dp的思路非常地巧妙！我只能拍案叫绝！！！比较难想！需要反复刷才能记得住这种思路！！！

```c++
// time:O(n^2),最坏情况下是遍历了2层长度为n的数组！
// space:O(n),使用辅助dp数组空间 且 长度为n
class Solution {
public:
    int LIS(vector<int>& arr) {
        //设置数组长度大小的动态规划辅助数组
        vector<int> dp(arr.size(), 1); 
        int res = 0;
        for(int i = 1; i < arr.size(); i++){
            for(int j = 0; j < i; j++){
                //可能j不是所需要的最大的，因此需要dp[i] < dp[j] + 1
                if(arr[i] > arr[j] && dp[i] < dp[j] + 1) {
                    // 因为此时判断出来下标为i为止的数组中是存在递增的子序列了！
                    // 那么长度可+1！即下标为i为止的数组中的最长递增的子序列长度是时候+1了！
                    //i点比j点大，理论上dp要加1
                    dp[i] = dp[j] + 1; 
                    //找到最大长度
                    res = max(res, dp[i]); // 更新结果
                }
            }
        }
        return res;
    }
};
// 这是根据我自己对于上面代码的理解写出来的版本：(我自己写的注释会更加利于自己的理解！)
// time:O(n^2),最坏情况下是遍历了2层长度为n的数组！
// space:O(n),使用辅助dp数组空间 且 长度为n
class Solution {
public:
    int LIS(vector<int>& arr) {
        int n = arr.size();
        // dp[i]:表示到达最终下标为i的子序列中严格上升子序列的最大长度为dp[i]
        vector<int> dp(n,1);
        // 数组中，最少都会有一个元素的子序列算是严格递增的！
        // so dp数组都初始化为1！
        if(n==0)return 0;// 先处理下特殊case！
        int res = 1;// 保存最长严格递增的子序列的长度！因为最少数组都会有一个元素，那么符合题意的子序列长度最少都是1了！
        for(int i = 1;i < n;++i){
            for(int j = 0;j < i;++j){
                // arr[j]有可能并不是使得dp[i]成为当前递增子序列长度最大的那个元素！
                // 比如：     0 1 2 3 4 5 6
                //    arr:  [6 3 1 5 2 3 7]
                //     dp:  [1 1 1 1 1 1 1]
                // 对于 arr[3] > arr[1] 且 dp[3] < dp[1] + 1,这表明arr[j]==arr[1]可以使得dp[i]成为最终下标到i为止的这个子序列中的递增子序列长度变为最大！==>dp[3] = dp[1]+1==2了！
                // 而   arr[3] > arr[2] 但 dp[3] == dp[1] + 1了，此时,这表明arr[j]==arr[2]无法使得dp[i]成为最终下标到i为止的这个子序列中的递增子序列长度变为最大！
                if(arr[i] > arr[j] && dp[i] < dp[j]+1){
                    dp[i] = dp[j]+1;// 严格递增子序列长度+1
                    res = max(res,dp[i]);// 更新最长严格递增的子序列的长度！
                }
            }
        }
        return res;
    }
};
```



#### ==<50>== [BM73-最长回文子串](https://www.nowcoder.com/practice/b4525d1d84934cf280439aeecc36f4af?tpId=295&tqId=25269&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

这个题目不论是牛客网还是leetcode上面的题目，我认为思路非常NB-PLUS!!!但是牛客网的官方题解会容易理解很多，但这都是基于我自己带入下面这个字符串画了完整的代码理解过程才能理解这种思路的！！！这道题目还是非常重要的！！！**（==不论是牛客还是leetcode上这两种提问方式我自己都必须要掌握！！！==）**

<img src="C:/Users/11602/Desktop/LeetCodeOfferNotes/git/%E7%AE%97%E6%B3%95%E5%AD%A6%E4%B9%A0%E6%88%AA%E5%9B%BE/3.jpg" alt="3" style="zoom:80%;" />

```c++
// 暴力法肯定就是2层for循环啦！
// 然后统计A[i,j]这个区间的回文子串最大长度 并更新结果res中保存的回文子串最大长度！
// 但显然这么do不合适！！！

// time:O(n^2),其中n是字符串A的长度,最坏情况下，遍历一轮字符串A中的各个字符后，每个单字符 or 每个双字符都需要扩展O(n)的时间复杂度
// space:O(1),没使用额外的辅助空间！
class Solution {
public:
    int getLongestPalindrome(string A) {
        int maxlen = 1;// 不论再怎么样，一个字符组成的字符串也属于回文串！
        // 以每个点为中心
        for(int i = 0;i < A.size()-1;++i){
            // 分奇数长度和偶数长度 向2边扩展
            maxlen = max({maxlen,middleExpand(A,i,i),middleExpand(A,i,i+1)});
        }
        return maxlen;
    }
    int middleExpand(const string& s,int begin,int end){
        // 每个中心点开始扩展
        while(begin >= 0 && end < s.size() && s[begin] == s[end]){
            begin--,end++;// 继续进行下一轮中心扩展法！
        }
        // 返回长度
        return end - begin - 1;
    }
};

// 下面这个版本是leetcode上面同一个题目的代码：（有一点点区别！）
// leetcode这里的提问是 求最长回文子串 这个字符串！
// ，而牛客网是         求最长回文子串的长度而已！
// so代码会有一些些的区别！
// time:O(n^2),n是字符串长度,最坏情况下在遍历整个str的每个单字符or双字符时，会有中心扩展O(n)的时间复杂度！
// space:O(1),没有使用额外的辅助空间！
class Solution {
private:
    pair<int,int> p;// 用来保存最终最长回文子串的[begin,end]下标位置！
    pair<int,int> tmp;// 用来临时保存最终最长回文子串的[begin,end]下标位置！
public:
    string longestPalindrome(string s) {
        int maxlen = 1;// 不论再怎么样，一个字符组成的字符串也属于回文串！
        p.first = p.second = 0;
        for(int i = 0;i < s.size()-1;++i){// 这里i < s.size - 1是为了防止中心扩展时越界！
            int t_odd  = middleExpand(s,i,i);// 计算奇数为中心的回文子串最长的长度！
            int t_even = middleExpand(s,i,i+1);// 计算偶数为中心的回文子串最长的长度！
            // 更新最长回文子串长度 的同时，更新该最长回文子串的[begin,end]下标位置！
            if(t_odd > maxlen && t_odd > t_even){
                maxlen = t_odd;
            }else if(t_even > maxlen && t_even > t_odd){
                maxlen = t_even;
            }
        }
        return s.substr(p.first,p.second-p.first+1);// 根据[begin,end]返回原string的最长回文子串！
    }
    int middleExpand(const string& s,int begin,int end){
        while(begin >= 0 && end < s.size() && s[begin] == s[end]){
            tmp.first = begin,tmp.second = end;// 记录最大的那个的下标！
            if(tmp.second - tmp.first > p.second - p.first){// 只保存最长的那个回文子串 的 [begin,end]下标位置！
                    p.first = tmp.first;
                    p.second = tmp.second;
            }
            begin--,end++;// 继续进行下一轮中心扩展法！
        }
        return end - begin - 1;// 返回使用中心扩展法获得的最长回文子串的长度！
    }
};
// leetcode way2:
class Solution {
public:
    pair<int,int> p;// 保存最终最长的回文子串的begin和end下标
    pair<int,int> tmp;// 保存临时最长回文子串的begin和end下标
    string longestPalindrome(string s) {
       int n = s.size();
       if(n==0)return "";// 先处理下特殊case
       for(int i = 0;i < n-1;++i){
           mid(s,i,i);// 以当前字符为中心进行奇数次扩展
           mid(s,i,i+1);// 以当前字符以及其右边的一个字符为中心，进行偶数次扩展！
           // 因为不止有 "aba" or "abcdcba"这种奇数次回文串！
           // 还会有 "abba" or "abccba" 这种偶数次的回文串！
       }
       return s.substr(p.first,p.second - p.first+1);
    }
    void mid(const string& s,int begin,int end){
        while(begin >= 0 && end < s.size() && s[begin] == s[end]){
            tmp.first = begin,tmp.second = end;
            if(p.second - p.first < tmp.second - tmp.first){
                p.second = tmp.second;
                p.first = tmp.first;
            }
            end++,begin--;// 继续下一轮找更长的回文子串！
        }
    }
};
```



#### ==<51>== [BM75-编辑距离(一) ](https://www.nowcoder.com/practice/6a1483b5be1547b1acd7940f867be0da?tpId=295&tqId=2294660&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

这个题目是==面试中非常常见并常考==的题目！！！我也是看了牛客官方的答案，并自己按照代码的意思画图后才知道能这么干的！

这个思路我觉得只要记忆下来就ok了，并不是很难写，但的确很难写出来！！！

<img src="C:/Users/11602/Desktop/LeetCodeOfferNotes/git/%E7%AE%97%E6%B3%95%E5%AD%A6%E4%B9%A0%E6%88%AA%E5%9B%BE/4.jpg" alt="4" style="zoom:80%;" />

```c++
// time:O(len1*len2+len1+len2)==O(len1*len2),主要是2层for循环遍历dp数组的耗时！
// space:O((len1+1)*(len2+1))==O(len1*len2),使用2维的dp辅助数组，大小是(len1+1)*(len2+1)
class Solution {
public:
    int editDistance(string str1, string str2) {
        int len1 = str1.size();
        int len2 = str2.size();
        // dp[i][j]:表示到str1[i]和str2[j]为止的子串需要的编辑距离就是dp[i][j]!
        vector<vector<int>> dp(len1+1,vector<int>(len2+1,0));
        // 初始化dp数组的边界！
        for(int i = 1;i <= len1;++i)dp[i][0] = dp[i-1][0]+1;
        for(int i = 1;i <= len2;++i)dp[0][i] = dp[0][i-1]+1;
        // 外for遍历第一个字符串的每个位置
        for(int i = 1;i <= len1;++i){
            // 内for遍历第二个字符串的每个位置
            for(int j = 1;j <= len2;++j){
                // 若是字符相同，则此处不用编辑了，因为相等了
                if(str1[i-1] == str2[j-1]){// 字符一样，无需任何的编辑操作，故等于其前一个对应的2个子串的编辑距离！
                    // 直接等于二者前一个的距离，相等了就不用do增删or改等编辑操作了！
                    dp[i][j] = dp[i-1][j-1];// 2字符相等时，不需额外增加编辑的操作了！
                }
                else{
                    // 此时2个字符串对应的字符不相等，但是，要想让字符相等，则既有可能是编辑str2[j],str1[i-1]不动，又或者是编辑str1[i],str2[j-1]不动，要么就是让str1[i-1],str2[j-1]的基础上，增加or删除一个字符就使得当前到str1的i下标为止，str2的j下标为止的字符串相等了！
                    // 改str1[i] or str2[j]
                    // 增删都是在str1[i-1],str2[j-1]的基础上do一个操作即可！
                    // so最后我们 选取最小的距离 加上此处编辑距离+1！因为不论是增删改（改有2种case）
                    dp[i][j] = min({dp[i-1][j-1],dp[i][j-1],dp[i-1][j]})+1;// 不论如何编辑，操作都是加1个的！
                }
            }
        }
        return dp[len1][len2];
    }
};
```



#### <52> [BM78-打家劫舍(一) ](https://www.nowcoder.com/practice/c5fbf7325fbd4c0ea3d0c3ea6bc6cc79?tpId=295&tqId=2285793&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

这打家劫舍问题，我就是do了N次都搞不懂搞不会！！！！没办法了，就是背思路吧！！！

<img src="C:/Users/11602/Desktop/LeetCodeOfferNotes/git/%E7%AE%97%E6%B3%95%E5%AD%A6%E4%B9%A0%E6%88%AA%E5%9B%BE/5.jpg" alt="5" style="zoom: 50%;" />

```c++
// time:O(n),遍历了一遍nums数组，其中n是其长度
// space:O(n),使用了辅助dp数组空间，长度就是n
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        // dp[i]:表示长度为i的数组，最多能够偷取dp[i]元！
        vector<int> dp(n+1,0);
        dp[0] = 0;// 偷长度为0的数组，你没法偷，so利益是0元！
        dp[1] = nums[0];// 长度为1只能偷第一家，此时收益最大！
        for(int i = 2;i <= n;++i){
            // 这里的i在dp中为数组长度，在nums中为下标。因此当计算本轮人家能够偷的钱时，要用nums[i-1]!而dp不用-1！这是非常key 的一个点！
            // 对于每个人家，可选择偷or不偷
            // 当选择偷本人家时，当前收益 dp[i] = dp[i-2] + nums[i-1];因为不能偷相邻人家的钱，因此必须要是偷完上上家所获得的钱+偷本人家所获得的钱才算当前收益！
            // 当选择不偷本人家时，当前收益 dp[i] = dp[i-1];// 因为不能偷相邻人家的钱，因此本轮收益==偷完上一个人家后所获得的收益！
            // 然后因为要获得最大的收益，因此上面2种case取max即可！
            dp[i] = max(dp[i-1],dp[i-2]+nums[i-1]);
        }
        return dp[n];
    }
};
```



#### <53> [BM78-打家劫舍(二) ](https://www.nowcoder.com/practice/a5c127769dd74a63ada7bff37d9c5815?tpId=295&tqId=2285837&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

这个题目是打家劫舍（一）的小升级版（每户人家都处于一个环形圈圈内），这个思路确实难想，但是牛客网答案是真的非常清晰！把它背下来也不是不行的！！！

```c++
// time:O(n*2)==O(n),遍历了nums数组2次！
// space:O(n),使用了辅助dp数组，长度是n
class Solution {
public:
    int rob(vector<int>& nums) {
        /* 
            主要思路：
            因为每户人家都处于一个环形圈圈中，这会造成第一户人家和最后一户人家是相邻的！
            因此，要是偷了第一户人家就一定不能偷最后户人家（因为相邻）
            同理，要是投了最后一户人家，就一定不能偷第一户人家（因为相邻）
            知道了这题只有这2种分类讨论的cases后，剩下的思路就和打家劫舍（一）没有啥区别的了！
        */
        int n = nums.size();
        // dp[i]:表示对于长度为i的数组nums，最多可以偷到dp[i]元！
        vector<int> dp(n+1,0);
        dp[0] = 0,dp[1] = nums[0];
        // 1-先偷第一户人家，不偷最后一户人家
        for(int i = 2;i < n;++i){// 这里不能=n是因为不能够偷最后一户人家的钱！
            dp[i] = max(dp[i-1],dp[i-2]+nums[i-1]);
        }
        int res1 = dp[n-1];
        // 2-再偷最后一户人家，不偷第一户人家
        dp.clear();// 清除dp的原数据
        dp[0] = 0,dp[1] = 0;// 既然不偷，那获得的收益肯定是0元啦！
        for(int i = 2;i <= n;++i){// 这里能=n是因为能够偷最后一户人家的钱！
            dp[i] = max(dp[i-1],dp[i-2]+nums[i-1]);
        }
        int res2 = dp[n];
        // 这2种cases取max的dp[i]返回即可！
        return max(res1,res2);
    }
};
```



#### <54> [BM80-买卖股票的最好时机(一)](https://www.nowcoder.com/practice/64b4262d4e6d4f6181cd45446a5821ec?tpId=295&tqId=625&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

在carl哥的网站上do过很多次了，但是就是记不住思路！！！       

```c++
// time:O(n),其中n是prices数组的长度
// space:O(n*2)==O(n),使用了辅助数组空间dp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        // dp[i][0]:表示到第i天为止，持有股票所得最大利润
        // dp[i][1]:表示到第i天为止，不持有股票所得最大利润
        vector<vector<int>> dp(n,vector<int>(2,0));
        dp[0][0] = -prices[0];
        dp[0][1] = 0;
        for(int i=1;i<n;++i){
            dp[i][0] =max(dp[i-1][0],0-prices[i]);// 因为买入股票之前必须要卖出才能够继续买入，因此必须从0开始计算当前持有的股票所获得最大利益
            dp[i][1] = max(dp[i-1][1],dp[i-1][0]+prices[i]);// 卖出股票时，必须这么计算：昨天持有股票要+上今天卖出股票后所获得最大利益,要么不卖出，那就是dp[i-1][1]
        }
        return dp[n-1][1];
    }
};
```







#### <55> [BM81-买卖股票的最好时机(二)](https://www.nowcoder.com/practice/9e5e3c2603064829b0a0bbfca10594e9?tpId=295&tqId=1073471&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

在carl哥的网站上do过很多次了，但是就是记不住思路！！！       

这道题目与买卖股票的最好时机(一)的区别是：本题总共可以买卖多次

```c++
// time:O(n),n是prices数组的长度！
// space:O(n)
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        // write code here
        int n=prices.size();
        vector<vector<int>> dp(n,vector<int>(2,0));
        dp[0][0]=-prices[0];// 第0天持有股票所得最大利润
        dp[0][1]=0;// 第0天不持有股票所得最大利润
        for(int i=1;i<n;++i){
            // 对于持有状态，1-可能前几天的持有的股票还没卖呢，就不能继续买入，则当前持有股票所得最大利益是dp[i-1][0]
            // 对于持有状态，2-可能今天要买入股票，则当前持有股票所得最大利益是dp[i-1][1]=dp[i-1][1]-prices[i];我卖出去之后才能继续买入！
            // 同理：
            // 对于不持有状态，1-可能前几天的持有的股票已经卖了或者还没买，则当前持有股票所得最大利益是dp[i-1][1]
            // 对于不持有状态，2-可能今天就要卖出股票了，则当前持有股票所得最大利益就是dp[i-1][0]+prices[i];
            dp[i][0]=max(dp[i-1][0],dp[i-1][1]-prices[i]);
            dp[i][1]=max(dp[i-1][1],dp[i-1][0]+prices[i]);
        }
        return dp[n-1][1];
    }
};
```



## **牛客网  《剑指offer 》刷题笔记**

这里的题目都是我自己一刷时没do出来的（看题解） or do的有些磕磕绊绊需要想挺久才do出来的题目！！！（暗示我自己这些剑指offer的题目为肯定是需要2刷甚至3or4刷的！）



注意：**我个人认为太过于难的题目我就不总结了！**

### 目前累计总共有==《20》==道题：

虽然这个题目很简单，思路我也基本背下来了，但是还是不能把**双指针的细节**弄好！

#### <1> [JZ5-替换空格](https://www.nowcoder.com/practice/0e26e5551f2b489b9f58bc83aa4b6c68?tpId=265&tqId=39209&rp=1&ru=/exam/oj/ta&qru=/exam/oj/ta&sourceUrl=%2Fexam%2Foj%2Fta%3Fpage%3D1%26tpId%3D13%26type%3D265&difficulty=undefined&judgeStatus=undefined&tags=&title=)

```c++
class Solution {
public:
    string replaceSpace(string s) {
        // 这里我直接在原地进行操作！
        // time:O(n),space:O(1)
        int oldSize = s.size(),spaceSize = 0;
        for(char ch : s){
            if(ch == ' ')spaceSize++;
        }
        int newSize = oldSize + spaceSize * 2;
        s.resize(newSize);// 原地弄成新的字符串长度！
        // 双指针法！
        int index = newSize - 1;
        for(int i = oldSize-1;i>=0;--i){
            if(s[i]!=' '){
                s[index--] = s[i];
            }else{
                // 把空格子换为%20
                s[index] = '0';
                s[index-1] = '2';
                s[index-2] = '%';
                index-=3;
            }
        }
        return s;
    }
};
```



#### <2> [JZ8-二叉树的下一个节点](https://www.nowcoder.com/practice/9023a0c988684a53960365b889ceaf5e?tpId=265&tqId=39212&rp=1&ru=/exam/oj/ta&qru=/exam/oj/ta&sourceUrl=%2Fexam%2Foj%2Fta%3Fpage%3D1%26tpId%3D13%26type%3D265&difficulty=undefined&judgeStatus=undefined&tags=&title=)



这个题目这么简单我居然没do出来，我人都傻了！！！不应该啊lzf！！！

```c++
/*
struct TreeLinkNode {
    int val;
    struct TreeLinkNode *left;
    struct TreeLinkNode *right;
    struct TreeLinkNode *next;
    TreeLinkNode(int x) :val(x), left(NULL), right(NULL), next(NULL) {
        
    }
};
*/
// time:O(n*2)==O(n),最坏case下遍历2遍整棵树！
// space:O(n),使用了额外的辅助空间v来存储中序遍历后的各个节点！
class Solution {
private:
    vector<TreeLinkNode* > v;
    void middleTraversal(TreeLinkNode* node){
        if(node == nullptr)return;
        // 中序遍历是 左中右
        middleTraversal(node->left);
        v.push_back(node);
        middleTraversal(node->right);
    }
public:
    TreeLinkNode* GetNext(TreeLinkNode* pNode) {
        TreeLinkNode * node = pNode;
        while(node->next)node = node->next;// 先找到整棵树的根节点！这一步very key！
        // do中序遍历，按左中右的顺序来存储all的节点！！！
        middleTraversal(node);
        // 在中序节点数组中找到该节点的下一个节点即可!
        for(int i=0;i < v.size();++i){
            if(v[i] == pNode && i + 1< v.size())return v[i+1];
        }
        return nullptr;
    }
};
```





#### <3> [JZ12-矩阵中的路径](https://www.nowcoder.com/practice/2a49359695a544b8939c77358d29b7e6?tpId=265&tqId=39216&rp=1&ru=/exam/oj/ta&qru=/exam/oj/ta&sourceUrl=%2Fexam%2Foj%2Fta%3FtpId%3D13&difficulty=undefined&judgeStatus=undefined&tags=&title=)

这个题目还真的是蛮难的！考记忆化**树形**深搜dfs！这个思路不容易想的！！！背下来！！！多copy即可！（画个图就能理解下面的算法思路了！不画图光用脑子想肯定是没法理解的！！！）

<img src="C:/Users/11602/Desktop/LeetCodeOfferNotes/git/%E7%AE%97%E6%B3%95%E5%AD%A6%E4%B9%A0%E6%88%AA%E5%9B%BE/7.jpg" alt="7" style="zoom: 50%;" />

```c++
class Solution {
public:
    bool hasPath(vector<vector<char> >& matrix, string word) {
        // 这个题目很明显实在 考察 记忆化dfs深搜索这个知识点！
        // 优先处理特殊情况
        if(matrix.size() == 0)return false;
        int row = matrix.size();// 行数
        int col = matrix[0].size();// 列数
        // 初始化flag矩阵记录看是否走过！
        vector< vector<bool> > flag(row,vector<bool>(col,false));
        // 遍历矩阵找起点
        for(int i=0;i<row;++i){
            for(int j=0;j<col;++j){
                // 通过dfs找到路径
                if(dfs(matrix,row,col,i,j,word,0,flag))return true;
            }
        } 
        return false;
    }
    bool dfs(vector<vector<char>>& matrix,int row,int col,int i,int j,string word,int k,vector<vector<bool>>& flag){
        if(i < 0 || i >= row || j < 0 || j >= col || (matrix[i][j] != word[k]) || (flag[i][j] == true) ){
            // 但凡是下标越界，字符不匹配，已经遍历过的都不能重复！
            return false;
        }
        // k为记录当前第几个字符
        if(k == word.size() - 1)return true;
        flag[i][j] = true;
        // 该节点向任意方向可行就可
        if  ( 
            dfs(matrix,row,col,i-1,j,word,k+1,flag) ||  
            dfs(matrix,row,col,i+1,j,word,k+1,flag) ||
            dfs(matrix,row,col,i,j-1,word,k+1,flag) ||
            dfs(matrix,row,col,i,j+1,word,k+1,flag)   
            ){return true;}
        // 经过此格子的此路径不符合题目要求，so回溯，让此格子就未被占用
        flag[i][j] = false;// ==> 回溯！
        return false;
    }
};

// 我自己理解了上面的代码之后，自己写的版本:
class Solution {
public:
    bool hasPath(vector<vector<char> >& matrix, string word) {
        // 这个题目需要用到一种叫做，记忆化-树形深搜dfs的算法思路！
        // 先处理下特殊的case
        if(matrix.size() == 0)return false;// 一旦居正大小为0，则返回false！
        int n = matrix.size(),m=matrix[0].size();
        vector<vector<bool>> flag(n,vector<bool>(m,false));
        // flag是标记二维数组，大小和矩阵一样！
        // 用来标记matrix矩阵中哪个字符已经走过了！就不能够再走了！
        for(int i=0;i<n;++i){
            for(int j=0;j<m;++j){
                if(dfs(matrix,word,n,m,i,j,0,flag))return true;
                // 因为没法确定矩阵中的哪个字符开头能够找到word的路径
                // so遍历矩阵中的每个字符来找word的开头
                // 并且继续树形dfs下去！找到这样一条路径就return true即可了！
            }
        }
        return false;
    }
    bool dfs(vector<vector<char>>& matrix,string word,int n,int m,int i,int j,int k,vector<vector<bool>> & flag){
        // 先处理下越界的情况！
        if(i<0 || i>=n || j<0||j>=m||(matrix[i][j] != word[k]) || flag[i][j] == true){
            return false;

            // 越界or当前矩阵元素与word中的对应字符不同 or当前字符已经走过了，用过了，就返回false！表示不存在这样一条路径==word！
        }
        // 判断是否该路径组成的字符串已经==word了！
        if(k == word.size() - 1)return true;
        // if没达到word，就继续记忆化树形dfs深搜下去即可！
        flag[i][j] = true;
        if(
            dfs(matrix,word,n,m,i-1,j,k+1,flag)||// 上
            dfs(matrix,word,n,m,i+1,j,k+1,flag)||// 下
            dfs(matrix,word,n,m,i,j-1,k+1,flag)||// 左
            dfs(matrix,word,n,m,i,j+1,k+1,flag)  // 右
        ){
            return true;
        }
        // 达成不了word,就不走该字符这条路了！
        flag[i][j] = false;// ==>相当于回溯的操作！
        return false;
    }
};
```





#### <4> [JZ31-栈的压入、弹出序列](https://www.nowcoder.com/practice/d77d11405cc7470d82554cb392585106?tpId=265&tqId=39233&rp=1&ru=/exam/oj/ta&qru=/exam/oj/ta&sourceUrl=%2Fexam%2Foj%2Fta%3FtpId%3D13&difficulty=undefined&judgeStatus=undefined&tags=&title=)

这个题目我好像在leetcode上面do过了的！解题的思路有点难度的！我看了牛客官网的answer后还思考画图了一会儿，确实不简单！

推荐使用题目给出的例子：pushV:[1,2,3,4,5];popV:[4,5,3,2,1]来画个图你就能够理解这个代码了！

```c++
// time:O(n),n是push or pop数组的长度,遍历了整个push or pop数组
// space:O(n),使用了辅助数组空间stack，最坏情况下stack的长度会达到n！
class Solution {
public:
    // 题目思路主要是：要检查 弹出序列popV 是否合法！
    // 一个栈的压入顺序pushV，可以对应多种 弹出序列popV
    // 因此重点就是要比较栈的弹出序列，因此才有了代码中for循环遍历弹出序列来do比较的操作！
    bool IsPopOrder(vector<int> pushV,vector<int> popV) {
        int pushSize = pushV.size();
        stack<int> stk;// 辅助栈空间！
        int in = 0;// 入栈下标！
        // for循环遍历弹出序列 检查其是否合法！
        for(int out = 0;out < popV.size();++out){
            while(in < pushSize && (stk.empty() || stk.top() != popV[out])){
                stk.push(pushV[in++]);// 压入元素到辅助栈空间中去！
            }
            if(stk.top() == popV[out]){
                stk.pop();
            }else return false;// 只要有不等的，就马上返回fasle！表示不可能是这个弹出序列popV！也即该弹出序列不合法！
        }
        return true;// 能够经得住for循环的“洗礼“,那自然就是一个合法的弹出序列了！
    }
};
```



#### <5> [JZ26-树的子结构](https://www.nowcoder.com/practice/6e196c44c7004d15b1610b9afca8bd88?tpId=265&tqId=39228&rp=1&ru=/exam/oj/ta&qru=/exam/oj/ta&sourceUrl=%2Fexam%2Foj%2Fta%3FtpId%3D13&difficulty=undefined&judgeStatus=undefined&tags=&title=)

这个题目在leetcode上的剑指offer上面我已经do过一次了！但在这里还是do不出来！！！这个思路还是在leetcode高赞评论区答案里面学来的，真的不太好想呀！！！只有背下来了！

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
// 学的评论区的高赞答案写的！！！太NB了！
// 可太难想了这个思路！
/*
    思路：1-先在树A中找到头结点 == 树B头结点的节点
         2-找到之后再将此时的A子结构的left和right 与 B的left和right逐个do队比，即可判断A是否含有B这样的子结构了！
*/
class Solution {
public:
    bool isSubStructure(TreeNode* A, TreeNode* B) {
        if(A == NULL || B == NULL) return false;//如果A到底了 或 B为空 则没有匹配到
        if(A->val == B->val)//找到可能的开头
        {
            if(detection(A, B)) return true;//伴随判断,找到了直接返回
        }
        return isSubStructure(A->left, B) || isSubStructure(A->right, B);//继续寻找可能的开头
    }
    bool detection(TreeNode* A, TreeNode* B)//根据B开始向下匹配
    {
        if(B == NULL) return true;//匹配到头了，直接返回true
        if(A == NULL || A->val != B->val) return false;//B有值,A没有 或者 A B值不同 返回假
        return detection(A->left, B->left) && detection(A->right, B->right);//向下匹配 见假为假
    }
};

// 这是我理解了上面的代码后写出来的版本：
class Solution {
public:
	// 好像在leetcode上面已经do过了！
    bool HasSubtree(TreeNode* A, TreeNode* B) {
       // 主要思路是：先在A中找到与B这棵树的头结点相同的节点
	   // 然后在此基础上，再逐个逐个让A的left与B的left，A的right和B的right来do对比，如果都一样就说明是子结构！否则就不是！
	   if(A == nullptr || B == nullptr)return false;// 这里题目规定了空树不是任何一棵树的子结构！
	   if(A->val == B->val){
		   if(traversal(A,B))return true;
	   }
	   return HasSubtree(A->left,B) || HasSubtree(A->right,B);
    }
	bool traversal(TreeNode* A,TreeNode* B){
		if(B == nullptr)return true;// if B遍历到空节点了说明B这个子结构就存在于A中！
		// if A遍历到空节点但B还有节点 or A节点的值不等于B对应节点的值，此时就说明B不是A的子结构！
		if(A == nullptr || A->val != B->val)return false;// 中
		// 然后如果A和B对应节点不为空且value一样的case下，就再继续递归判断下A的left right 与B对应的left和right是否都一样！
		bool left = traversal(A->left,B->left);// 左
		bool right = traversal(A->right,B->right);// 右
		return left && right;
	}
};
```







#### ==<6>== [JZ33-二叉搜索树的后序遍历序列](https://www.nowcoder.com/practice/a861533d45854474ac791d90e447bafd?tpId=265&tqId=39235&rp=1&ru=/exam/oj/ta&qru=/exam/oj/ta&sourceUrl=%2Fexam%2Foj%2Fta%3FtpId%3D13&difficulty=undefined&judgeStatus=undefined&tags=&title=)

这个题目真的难！！！我在leetcode上面原来已经do过一次的了！但是看牛客官方的非递归的答案我看不懂！！！最后还是看B站一个小姐姐up讲解的剑指offer才明白的！！！（用到了分治递归的思想！！！）

[LeetCode力扣刷题 | 剑指Offer 33. 二叉搜索树的后序遍历序列](https://www.bilibili.com/video/BV1FK4y1j73k/?spm_id_from=333.337.search-card.all.click&vd_source=b050ab0adaffa51f5ad24d77efa40057)



<img src="C:/Users/11602/Desktop/LeetCodeOfferNotes/git/%E7%AE%97%E6%B3%95%E5%AD%A6%E4%B9%A0%E6%88%AA%E5%9B%BE/8.JPG" alt="8" style="zoom:50%;" />



<img src="C:/Users/11602/Desktop/LeetCodeOfferNotes/git/%E7%AE%97%E6%B3%95%E5%AD%A6%E4%B9%A0%E6%88%AA%E5%9B%BE/9.JPG" alt="9" style="zoom: 80%;" />

```c++
class Solution {
public:
    // 看了评论区“递归和栈两种方式解决，最好的击败了100%”的用户这个标题的大佬才do出来的！
    // 分治递归法：(左闭右闭区间)
    // 思路：对于BST来说，后序遍历后最后一个元素必然是根节点root
    // 然后后续遍历的第一个大于root的节点到root的前一个节点这个区间内的节点必然为BST的右子树
    // 然后从前面到第一个大于root的节点的前一个节点的区间必然是BST的左子树
    // 即：[left,第一个大于当前root节点-1]是BST的左子树区间
    //    [第一个大于当前root节点,root节点-1]是BST的右子树区间
    // 若不满足以上的2个条件的话，就一定不是BST了！
    bool traversal(const vector<int>& postorder,int left,int right){
        /*
            当left == right时说明此时分治到只剩下一个节点值了，就没必要继续分治了！直接认定是BST
            当left > right时说明此时分治到越了数组的界了，就没必要继续了！直接认定是BST
        */
        if(left >= right)return true;
        int fenGeIdx = left;// 分割点
        int curRoot = postorder[right];// 取得本轮递归根节点值
        while(postorder[fenGeIdx] < curRoot)fenGeIdx++;
        int tmp = fenGeIdx;
        /*
            因为要确认当前的[第一个大于当前root节点,root节点-1]区间必须都要大于curRoot值
            才能确定是BST当前根节点curRoot的右子树！
            若有小于curRoot值的就马上return false表示不是BST了！
        */
        while(tmp < right){
            if(postorder[tmp++] < curRoot)return false;
        }
        tmp = fenGeIdx-1;
        /*
            同上理，要确定[left,第一个大于当前root节点-1]是BST的左子树区间这个区间的all节点值
            都要小于当前根节点curRoot的值！才符合BST！
        */
        while(tmp >= left){
            if(postorder[tmp--] > curRoot)return false;
        }
        return traversal(postorder,left,fenGeIdx-1) && traversal(postorder,fenGeIdx,right-1);
    }
    bool VerifySquenceOfBST(vector<int>& postorder) {
        // BST的中序遍历一定是升序的(从小到大的)！
        // 这个特性是一定要用上的！
        return traversal(postorder,0,postorder.size()-1);
    }
};
// 下面是我理解了上面的代码后写出来的版本：
// 本题用到了分治递归的思想！
/*
    一个BST，其后序遍历到last一个元素必然是根节点元素
    然后从前面到第一个大于当前last节点元素值的前一个元素这个区间必然是BST的左子树区间
    从大于当前last节点元素值的第一个元素到last的前一个元素这个区间必然是BST的右子树区间
    if 不满足上面这2个条件，那么就一定不是BST的后序遍历序列了！        
*/
class Solution {
public:
    bool verifyPostorder(vector<int>& postorder) {
        return traversal(postorder,0,postorder.size()-1);// 左闭右闭区间！
    }
    bool traversal(const vector<int>& p,int l,int r){
        if(l >= r)return true;// 剩下一个元素说明满足条件！
        // 先找到第一个大于p[r]这个last节点元素值的第一个元素值do为分割点！
        // 这个过程相当于是先判断当前的左子树区间是否合法（符合BST的规则）
        // 当前的root->val == p[r]!
        int fenGeIdx = l;
        while(p[fenGeIdx] < p[r])fenGeIdx++;// 左子树节点值必然小于root->val
        // 然后再判断当前的右子树区间是否合法（符合BST的规则）
        int i = fenGeIdx;
        while(p[i] > p[r])i++;// 右子树节点值必然大于root->val
        // 走到这里就说明当前的左右子区间符合BST的规则！
        // 就继续分治递归左右子区间看是否还符合BST的规则！（大问题分解为小问题,so可递归）
        bool leftMeet = traversal(p,l,fenGeIdx-1);
        bool rightMeet = traversal(p,fenGeIdx,r-1);
        // 当然还需要判断是否符合i==r，如果都到不了r说明根本就不是一个正确的BST的后序遍历到序列！
        return i == r && leftMeet && rightMeet;
    }
};

// 简短不带注释版本：
class Solution {
public:
    bool verifyPostorder(vector<int>& postorder) {
        return traversal(postorder,0,postorder.size()-1);// 左闭右闭区间！
    }
    bool traversal(const vector<int>& p,int l,int r){
        if(l >= r)return true;
        int fenGeIdx = l;
        while(p[fenGeIdx] < p[r])fenGeIdx++;
        int i = fenGeIdx;
        while(p[i] > p[r])i++;
        return i == r && traversal(p,l,fenGeIdx-1) && traversal(p,fenGeIdx,r-1);
    }
};
```



#### <7> [JZ34-二叉树中和为某一值的路径（二）](https://www.nowcoder.com/practice/b736e784e3e34731af99065031301bca?tpId=265&tqId=39236&rp=1&ru=/exam/oj/ta&qru=/exam/oj/ta&sourceUrl=%2Fexam%2Foj%2Fta%3FtpId%3D13&difficulty=undefined&judgeStatus=undefined&tags=&title=)

这个题目命名在leetcode上面已经do得很熟悉了，但是就是在牛客上没写好思路，思路不完整！！！

```c++
class Solution {
private:
	vector<vector<int>> res;// 保存all路径的结果
	vector<int> tmp;// 临时保存一条路径的结果
public:
	// 这个题目leetcode上面do过很多次了！
	// 因为要找all的路径，因此必须要遍历整棵树一遍！回溯法！且不需要递归的返回值！
	void traversal(TreeNode* cur,int tar){
		tmp.push_back(cur->val);// 不管3721先把当前节点加入到临时路径结果tmp中！
		if(!cur->left && !cur->right){
			// 只有 遍历到根节点才算是一条有效的路径！(因为题目就是要找完整的从根节点到叶子节点的路径！)
			if(tar < 0)return;
			else if(tar == 0){
				res.push_back(tmp);// 保存结果
				return;
			}
		}
		// 左不空就继续向左子树遍历找从根节点到叶子的路径！
		if(cur->left){
			tar -= cur->left->val;
			traversal(cur->left,tar);
			tar += cur->left->val;// 回溯！
			tmp.pop_back();
		}
		// 右不空就继续向右子树遍历找从根节点到叶子的路径！
		if(cur->right){
			tar -= cur->right->val;
			traversal(cur->right,tar);
			tar += cur->right->val;// 回溯！
			tmp.pop_back();
		}
	}
    vector<vector<int>> FindPath(TreeNode* root,int expectNumber) {
		if(root==nullptr)return {};// 先处理特殊case！
		res.clear();
		tmp.clear();
		traversal(root,expectNumber-root->val);
		return res;
    }
};

// 版本2：（我个人觉得这样子更好理解，画个图就能理解了！）
class Solution {
private:
	vector<vector<int>> res;
	vector<int> tmp;
public:
	void traversal(TreeNode* cur,int tar){
		tmp.push_back(cur->val);
		if(!cur->left && !cur->right){
			if(tar - cur->val == 0)res.push_back(tmp);	
			return;
		}
		if(cur->left){
			traversal(cur->left,tar - cur->val);
			tmp.pop_back();
		}
		if(cur->right){
			traversal(cur->right,tar - cur->val);
			tmp.pop_back();
		}
	}
    vector<vector<int>> FindPath(TreeNode* root,int expectNumber) {
		if(root == nullptr)return {};
		res.clear();tmp.clear();
		traversal(root,expectNumber);
		return res;
    }
};
```





#### <8> [JZ50-第一个只出现一次的字符](https://www.nowcoder.com/practice/1c82e8cf713b4bbeb2a5b31cf5b0417c?tpId=265&tqId=39248&rp=1&ru=/exam/oj/ta&qru=/exam/oj/ta&sourceUrl=%2Fexam%2Foj%2Fta%3FtpId%3D13&difficulty=undefined&judgeStatus=undefined&tags=&title=)

这么简单的题目我都没do出来！我人傻了都！！！

```c++
// time:O(n*2)==O(n),n是字符串的长度,单独遍历了2次整个字符串！
// space:O(1),虽说使用了哈希表的额外空间，但是该哈希表只会存放字母！即便是算上大小写字母加起来都只有52种！因此O(52)==O(1),是常量级的空间！
class Solution {
public:
    int FirstNotRepeatingChar(string str) {
        // 哈希表
        unordered_map<char,int> hmap;
        for(char& ch : str)hmap[ch]++;
        for(int i = 0;i < str.size();++i){
            if(hmap[str[i]] == 1)return i;
        }
        return -1;
    }
};
```



#### <9> [JZ82-二叉树中和为某一值的路径（一）](https://www.nowcoder.com/practice/508378c0823c423baa723ce448cbfd0c?tpId=265&tqId=39278&rp=1&ru=/exam/oj/ta&qru=/exam/oj/ta&sourceUrl=%2Fexam%2Foj%2Fta%3Fpage%3D2%26tpId%3D13%26type%3D265&difficulty=undefined&judgeStatus=undefined&tags=&title=)

这个题目和<7> 差不多！不一定需要遍历整棵二叉树！只要找到一条从根节点到叶子节点的路径，该路径上的节点值的和==tar即可了！

```c++
// 版本1:（官方answer）
class Solution {
public:
    bool traversal(TreeNode* cur,int tar){
        // 因为找到从根节点到叶子节点路径上节点值的和==tar的就直接返回即可
        // 不一定需要遍历完整棵二叉树！
        if(!cur->left && !cur->right){
            if(tar == 0)return true;
            return false;
        }
        if(cur->left){
            if(traversal(cur->left,tar-cur->left->val))return true;
        }
        if(cur->right){
            if(traversal(cur->right,tar-cur->right->val))return true;
        }
        return false;
    }
    bool hasPathSum(TreeNode* root, int sum) {
        if(root == nullptr)return false;
        return traversal(root,sum-root->val);
    }
};

// 版本2：（我自己认为这样写更容易理解，具体直接拿题目的例子来画个图就ok的了！）
class Solution {
public:
    bool traversal(TreeNode* cur,int tar){
        if(!cur->left && !cur->right){
            if(tar - cur->val == 0)return true;
            return false;
        }
        if(cur->left){
            if(traversal(cur->left,tar - cur->val))return true;
        }
        if(cur->right){
            if(traversal(cur->right,tar - cur->val))return true;
        }
        return false;
    }
    bool hasPathSum(TreeNode* root, int sum) {
        if(root == nullptr)return false;
        return traversal(root,sum);
    }
};
```





#### <10> [JZ82-二叉树中和为某一值的路径（三）](https://www.nowcoder.com/practice/965fef32cae14a17a8e86c76ffe3131f?tpId=265&tqId=39291&rp=1&ru=/exam/oj/ta&qru=/exam/oj/ta&sourceUrl=%2Fexam%2Foj%2Fta%3Fpage%3D2%26tpId%3D13%26type%3D265&difficulty=undefined&judgeStatus=undefined&tags=&title=)

这个题目和之前两个二叉树某路径和的题目有一点点不一样，so应该这么do！！！

```c++
/**
 * struct TreeNode {
 *	int val;
 *	struct TreeNode *left;
 *	struct TreeNode *right;
 *	TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 * };
 */
class Solution {
public:
    int res = 0;
    int FindPath(TreeNode* root, int sum) {
        // write code here
        if(root == nullptr)return res;// 其实返回什么在这里并不重要！
        // 因为不知道以哪个节点为树的根节点时能够找到路径和==sun的路径
        // 因此从上到下-前序遍历一次这棵树来搜这样符合题目意思的路径！
        traversal(root,sum);// 中,看以当前节点为根节点是树是否能找到这样的路径！
        FindPath(root->left,sum);// 左,看以当前节点的left节点为根节点是树是否能找到这样的路径！
        FindPath(root->right,sum);// 右,看以当前节点的right节点为根节点是树是否能找到这样的路径！
        return res;// 最后返回结果！
    }
    void traversal(TreeNode* cur,int tar){
        // if(tar - cur->val < 0 || cur==nullptr)return;
        if(cur==nullptr)return;// 递归函数的终止条件！
        // 中 左 右
        if(tar - cur->val == 0)res++;
        // 继续往左右子节点来看是否能够再找到这样的路径！
        traversal(cur->left,tar - cur->val);
        traversal(cur->right,tar - cur->val);
    }
};
```





#### <11> [JZ76-删除链表中重复的节点](https://www.nowcoder.com/practice/fc533c45b73a41b0b44ccba763f866ef?tpId=265&tqId=39270&rp=1&ru=/exam/oj/ta&qru=/exam/oj/ta&sourceUrl=%2Fexam%2Foj%2Fta%3Fpage%3D2%26tpId%3D13%26type%3D265&difficulty=undefined&judgeStatus=undefined&tags=&title=)

这个题目我在leetcode上面do过了，但是就是忘记思路了！！！看的牛客官方的题解才do出来的！

```c++
// time:O(n),n是list节点总数，只遍历了一遍all的节点！
// space:O(1),没使用额外辅助空间，只开辟了临时指针变量，是常量级空间
class Solution {
public:
    ListNode* deleteDuplication(ListNode* pHead) {
        if(pHead == nullptr)return pHead;// 处理特殊case
        ListNode* dummyNode = new ListNode(-1);
        dummyNode->next = pHead;
        ListNode* cur = dummyNode;
        while(cur->next && cur->next->next){
            if(cur->next->val == cur->next->next->val){
                int tmp = cur->next->val;
                while(cur->next && cur->next->val == tmp){
                    cur->next = cur->next->next;// 去重！跳过重复的节点
                }
            }else{
                cur = cur->next;
            }
        }
        return dummyNode->next;
    }
};
```





#### <12> [JZ81-调整数组顺序使奇数位于偶数前面（二）](https://www.nowcoder.com/practice/0c1b486d987b4269b398fee374584fc8?tpId=265&tqId=39274&rp=1&ru=/exam/oj/ta&qru=/exam/oj/ta&sourceUrl=%2Fexam%2Foj%2Fta%3Fpage%3D2%26tpId%3D13%26type%3D265&difficulty=undefined&judgeStatus=undefined&tags=&title=)

个人认为我自己写的代码比牛客网官方的answer好！！！

```c++
// time:O(n),n是原数组长度，遍历了原数组各个元素一遍！
// sapce:O(1),没有使用额外的辅助空间！
class Solution {
public:
    // 这个题目不使用额外空间才是真正难的地方！
    vector<int> reOrderArrayTwo(vector<int>& array) {
        // 先处理特殊case
        if(array.size() == 0)return {};
        // 奇数放前面，偶数放后面
        int odd = 0;// 奇数的idx
        for(int i = 0;i < array.size();++i){
            if(array[i] % 2 == 1){// 奇数
                std::swap(array[odd++],array[i]);
            }
            // 偶数 不do任何事情 
        }
        return array;
    }
};
```







#### <13> [JZ54-二叉搜索树的第k个节点](https://www.nowcoder.com/practice/57aa0bab91884a10b5136ca2c087f8ff?tpId=265&tqId=39251&rp=1&ru=/exam/oj/ta&qru=/exam/oj/ta&sourceUrl=%2Fexam%2Foj%2Fta%3Fpage%3D2%26tpId%3D13%26type%3D265&difficulty=undefined&judgeStatus=undefined&tags=&title=)

这个题目我自己do出来了，不难的，但没有牛客官网的答案那么好（我个人认为）

```c++
// way1:(我自己写的answer)

// time:O(n),n是BT的节点总数，all的节点都遍历了一遍了！
// space:O(n),栈空间的最大深度是二叉树退化为list的时候，此时栈空间最多占用O(n)
class Solution {
public:
    vector<int> v;
    void mid(TreeNode* cur){
        if(cur==nullptr)return;
        // 左中右
        mid(cur->left);
        v.push_back(cur->val);
        mid(cur->right);
    }
    int KthNode(TreeNode* proot, int k) {
        // 先暴力法来deal问题！
        if(proot == nullptr)return -1;
        v.clear();
        mid(proot);
        // 当k不符合条件时也需要返回-1！
        if(k - 1 >= v.size())return -1;
        return v[k-1];
    }
};
// way2:(牛客官方的answer)

// time:O(n),n是BT的节点数，all的节点都遍历了一遍
// space:O(n),最坏情况下BST会退化为list，那么此时最多使用的栈空间就是O(n)

class Solution {
public:
    int cnt = 0;// 计数，看什么时候到达k个数！
    // 中序遍历到第k个数就是题目要求的数了！
    int res = -1;// 保存结果！
    void mid(TreeNode* cur,int k){
        if(cur==nullptr || cnt > k)return;
        // 左  中  右
        mid(cur->left,k);
        cnt++;
        if(cnt == k){
            res = cur->val;
            return;
        }
        mid(cur->right,k);
    }
    int KthNode(TreeNode* proot, int k) {
       // 思路：
       /*
        BST的中序遍历本来就是升序排序了的！
        利用这一特性，中序遍历BST的同时记录当前遍历到第几个节点来，就可以找到第k个节点，此时这个节点就是第k小的节点！符合题目意思！
       */
       if(proot==nullptr)return -1;
       mid(proot,k);
       return res;
    }
};
```





#### <14> [JZ53-数字在升序数组中出现的次数](https://www.nowcoder.com/practice/70610bf967994b22bb1c26f9ae901fa2?tpId=265&tqId=39266&rp=1&ru=/exam/oj/ta&qru=/exam/oj/ta&sourceUrl=%2Fexam%2Foj%2Fta%3Fpage%3D2%26tpId%3D13%26type%3D265&difficulty=undefined&judgeStatus=undefined&tags=&title=)

这个题目的思路是我见过的比较新奇的一个题！需要好好消化理解！！！拿 arr = [1 2 3 3 3 3 4 5]画个图就明白的了！

```c++
// time:O(logn),二分法
// space:O(1)
class Solution {
public:
    int GetNumberOfK(vector<int> data ,int k) {
        // time de 复杂度是O(logn)
        // 就是要考察二分法！
        // 主要logic思路是：
        // 返回出现k的左右边界的差，就是k出现的次数了！
        return binary_s(data,k+0.5)-binary_s(data,k-0.5);
    }
        int binary_s(vector<int>& data, float k){
        // 注意！这里必须是float or double
        // 不然的话没法把k+0.5传入进来do二分！！！
        int left = 0;
        int right = data.size() - 1;
        //二分左右界
        while(left <= right){
            int mid = (left + right) / 2;
            if(data[mid] < k)
                left = mid + 1;
            else if(data[mid] > k)
                right = mid - 1;
        }
        return left;
    }
};
```



#### <15> [JZ57-和为S的两个数字](https://www.nowcoder.com/practice/390da4f7a00f44bea7c2f3d19491311b?tpId=265&tqId=39254&rp=1&ru=/exam/oj/ta&qru=/exam/oj/ta&sourceUrl=%2Fexam%2Foj%2Fta%3Fpage%3D2%26tpId%3D13%26type%3D265&difficulty=undefined&judgeStatus=undefined&tags=&title=)

 这个题目我在leetcode上面do过了，当时根本没看答案，但是现在要看下答案才知道能够这么来do！

```c++
// way1:哈希表法

// time：O(n),n是数组长度！最坏情况下是遍历整个数组一次！
// sapce:O(n),哈希表最坏情况下占用数组空间是O(n)
class Solution {
public:
    vector<int> FindNumbersWithSum(vector<int> array,int sum) {
        // 先处理特殊case
        if(array.size()==0)return {};
        // 用哈希表来do！
        unordered_map<int,int> hashMap;
        for(int& num : array){
            int tmp = sum - num;
            if(hashMap.find(tmp)==hashMap.end()){
                hashMap[num]++;
            }else return {num,tmp};
        }
        return {};
    }
};
// way2:双指针法

// time：O(n),n是数组长度！最坏情况下是：左或者右指针遍历整个数组一次！
// sapce:O(1),常量级空间
class Solution {
public:
    vector<int> FindNumbersWithSum(vector<int> array,int sum) {
        // 先处理特殊case
        if(array.size()==0)return {};
        // 用双指针来do
        int l=0,r=array.size()-1;
        while(l<r){
            int tmp = array[l]+array[r];
            if(tmp < sum)l++;
            else if(tmp > sum)r--;
            else return {array[l],array[r]};
        }
        return {};
    }
};
```





#### <16> [JZ66-构建乘积数组](https://www.nowcoder.com/practice/94a4d381a68b47b7a8bed86f2975db46?tpId=265&tqId=39261&rp=1&ru=/exam/oj/ta&qru=/exam/oj/ta&sourceUrl=%2Fexam%2Foj%2Fta%3Fpage%3D2%26tpId%3D13%26type%3D265&difficulty=undefined&judgeStatus=undefined&tags=&title=)

这个题目我在leetcode上面do过好多次了，但是还是不记得思路！！！

```c++
// time:O(n*2)==O(n),遍历了数组A两次！
// space:O(1),除了数组res是必要的返回数组空间外，没有使用额外的辅助空间了！
// 思路：
class Solution {
public:
    vector<int> multiply(const vector<int>& A) {
        int size = A.size();
        if(size==0)return {};
        vector<int> res(size,1);
        for(int i=1;i<size;++i){
            res[i] = res[i-1]*A[i-1];
        }
        int tmp = 1;
        for(int i=size-1;i>=0;--i){
            res[i] *= tmp;
            tmp *= A[i];
        }
        return res;
    }
};
```



#### <17> [JZ45-把数组排成最小的数](https://www.nowcoder.com/practice/8fecd3f8ba334add803bf2a06af1b993?tpId=265&tqId=39246&rp=1&ru=/exam/oj/ta&qru=/exam/oj/ta&sourceUrl=%2Fexam%2Foj%2Fta%3FtpId%3D13&difficulty=undefined&judgeStatus=undefined&tags=&title=)

这个题目我用回溯法是能够do出来，但是题目测试用例有3个超时了，时间复杂度有点高！so直接看牛客网答案来学习思路！

这个题目之前在leetcode上面也见识过，但就是想不起来还能够这么do的！你拿numbers = [3,11]这个数组来验证一下这份代码就能完全明白了！ 

```c++
class Solution {
public:
    string PrintMinNumber(vector<int> numbers) {
        // 先处理特殊case
        int size = numbers.size();
        if(size==0)return "";
        vector<string> strs;
        for(int& num : numbers)strs.push_back(to_string(num));
        // 先do自定义排序，x+y < y+x (x,y都是字符串),只要满足该条件，最后排序出来的字符串数组就是数字和最小的字符串数组了！
        std::sort(strs.begin(),strs.end(),[](string& x,string& y){
            return x+y < y+x;
        });
        string res = "";// 保存结果字符串
        for(string str : strs)res += str;
        return res;
    }
};
```





#### <18> [JZ14-剪绳子](https://www.nowcoder.com/practice/57d85990ba5b440ab888fc72b0751bf8?tpId=265&tqId=39218&rp=1&ru=/exam/oj/ta&qru=/exam/oj/ta&sourceUrl=%2Fexam%2Foj%2Fta%3FtpId%3D13&difficulty=undefined&judgeStatus=undefined&tags=&title=)

真的搞不懂这个题目怎么do，我看了B站好几个讲解的视频还有题解，都搞不懂！！！直接背下来吧这个思路！（面试前多回顾我这份笔记即可了！）

```c++
// time:O(n*2),n是绳子原长度
// space:O(n+1)==O(n)
// 这个题目我怎么看题解和视频都看不懂，只能背下来了！！！
class Solution {
public:
    int cutRope(int n) {
        // m <= n
        if(n <= 3)return n-1;// 因为一定要剪成m段，而m是大于1的
        // 也即是必须至少剪为2段，也即至少剪一次的意思！
        vector<int> dp(n+1,0);
        dp[0] = 0;
        dp[1] = 1;// 长度为1的绳子剪完后最大乘积是1
        dp[2] = 2;// 长度为2的绳子剪完后最大乘积是2
        dp[3] = 3;// 长度为3的绳子剪完后最大乘积是3
        dp[4] = 4;// 长度为4的绳子剪完后最大乘积是4
        for(int i=5;i<=n;++i){
            for(int j=1;j<i;++j){
                // 要么不剪，要么剪成 长为(j) 和另一段 长为(i-j) 的子绳！然后取其乘积最大值即可！
                dp[i] = max(dp[i],dp[i-j]*j);
            }
        }
        return dp[n];
    }
};
```



#### <19> [JZ49-丑数](https://www.nowcoder.com/practice/6aa9e04fc3794f68acf8778237ba065b?tpId=265&tqId=39247&rp=1&ru=/exam/oj/ta&qru=/exam/oj/ta&sourceUrl=%2Fexam%2Foj%2Fta%3FtpId%3D13&difficulty=undefined&judgeStatus=undefined&tags=&title=)

这个题目说模拟题，但是思路确实很NB-PLUS的！！！牛客网的answer比leetcode上面的那些傻逼题解清晰多了也容易理解多了！

直接拿index==7，求第7个丑数来画图就能理解下面的这份代码了！（这个题目还是蛮不错的！）

```c++
class Solution {
public:
    int GetUglyNumber_Solution(int index) {
        //排除0
        if(index == 0)
            return 0; 
        //要乘的因数
        vector<int> factors = {2, 3, 5}; 
        //去重
        unordered_map<long, int> mp; 
        //小顶堆
        priority_queue<long, vector<long>, greater<long>> pq; 
        //1先进去
        mp[1] = 1; 
        pq.push(1);
        long res = 0;
        for(int i = 0; i < index; i++){ 
            //每次取最小的
            res = pq.top(); 
            pq.pop();
            for(int j = 0; j < 3; j++){
                //乘上因数
                long next = res * factors[j]; 
                //只取未出现过的
                if(mp.find(next) == mp.end()){  
                    mp[next] = 1;
                    pq.push(next);
                }
            }
        }
        return (int)res;
    }
};

// 下面的代码版本是我自己画图理解之后自己写出来的(这个自己理解过的代码带注释的版本才是真正属于我自己的题解)：
// 注意：因为题目的测试用例中的数字do了乘积后有超过int最大范围的整数，因此需要用long 来保存临时变量和结果的值！（以防止越界）
	// 解题思路：
    // 使用哈希表和优先级队列来do这个事情！
    // 因为all的丑数都是{2,3,5}的倍数！
    // 即丑数都是：{2*x,3*x,5*x}
    // 那么从小到大返回第n个丑数
    // 每一轮都抽取丑数集合中的最小值即可（这个可用优先级队列来实现，因为优先级队列的top()就能返回其中存放到all对象的最小or最大的那个）
class Solution {
public:
    int GetUglyNumber_Solution(int index) {// 求的是第index个丑数的值
        // 先排除特殊case 0
        if(index == 0)return 0;
        // 用优先队列 来记录丑数 
        priority_queue<long,vector<long>,greater<long>> pq;
        // 用hashMap来去重（给丑数去重）！
        unordered_map<long,int> hashMap;
        // 先把第一个丑数1push到上面这2种数据结构中
        pq.push(1);
        hashMap[1] = 1;
        long res = 0;// res变量用于保存第n个丑数
        vector<int> factors = {2,3,5};// 构成丑数的3个质因子！
        // 因为all的丑数都是由{2,3,5}的倍数构成
        // 即{2*x,3*x,5*x}都属于丑数，毫无例外！因此可以根据此规律来找第n个丑数！
        // 外for用于找第(i)个丑数
        for(int i = 1;i <= index;++i){
            res = pq.top();
            pq.pop();
            // 内for用于求新的一轮丑数，同时记录新丑数的值并do去重操作。
            for(int j = 0;j < 3;++j){
                long next = res * factors[j];// 求新一轮的丑数值
                if(hashMap.find(next)==hashMap.end()){// 记录新丑数到hashMap和priority_queue中！
                    // 找不到该丑数才push进来（去重）
                    hashMap[next] = 1;// 1表示出现一次的意思！就是记录该丑数出现了一下而已！没有什么特殊的含义！
                    pq.push(next);// 同时也push进优先级队列中
                }
            }
        }
        return int(res);
    }
};
```





#### <20> [JZ48-最长不含重复字符的子字符串](https://www.nowcoder.com/practice/48d2ff79b8564c40a50fa79f9d5fa9c7?tpId=265&tqId=39289&rp=1&ru=/exam/oj/ta&qru=/exam/oj/ta&sourceUrl=%2Fexam%2Foj%2Fta%3FtpId%3D13&difficulty=undefined&judgeStatus=undefined&tags=&title=)

这个题目我记得我在leetcode上面do了好多次了，但是我就是不记得解题思路！稍微拿string = "abcabcbb"; 来画个图你都能马上理解这个代码的！！！不难！

```c++
// time:O(n),n是原字符串长度
// sapce:O(n),使用了辅助的哈希表空间，最坏情况下整个原字符串都是不重复的，因此哈希表的长度就是字符串长度==n！
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        // 注意：这里求的是子串（必须要连续），而非子序列（可以不连续）
        int res = 0;// 保存结果
        // 思路：滑动窗口+哈希表（推荐使用）
        // 先来一个哈希表，用于统计字符元素出现的次数
        // 然后，使用双指针法维护一个滑动窗口，一旦出现重复（hashMap可判断该字符是否重复）就窗口左指针右移动，否则右指针右移动！
        // 简单来说解题思路是：用哈希表统计每个滑动窗口内构成的字符串的无重复子串的各个字符出现的次数，然后双指针维护一个滑动窗口！注意：右指针先移动，左指针只有出现重复时才更新！
        unordered_map<char,int> hm;
        for(int left = 0,right = 0;right < s.size();++right){
            // 先让窗口右指针右移动，并用哈希表统计当前窗口内对应的这个字符的出现次数！
            hm[s[right]]++;
            while(hm[s[right]] > 1){
                hm[s[left++]]--;// 窗口左指针右移动，且hm中统计的该字符的出现次数-1！
            }
            // 每一轮都统计最长不重复子字符串的长度！
            res = max(res,right-left+1);
        }
        return res;
    }
};
```





#### <21> [JZ48-]()

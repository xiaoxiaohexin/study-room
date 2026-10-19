export const ERROR_MESSAGES = {
	HTTP: {
		ACCESS_DENIED: '访问被拒绝',
		GENERIC: '请求失败',
		INTERNAL_ERROR: '服务端错误 - 请检查后端日志',
		NOT_FOUND: '未找到请求的资源',
		TEMPORARILY_UNAVAILABLE: '服务暂时不可用'
	},
	NETWORK: {
		GENERIC: '连接服务器失败',
		NXDOMAIN: '未找到服务器 - 请检查服务器地址',
		REFUSED: '连接被拒绝 - 服务器可能未启动',
		TIMEOUT: '请求超时',
		UNREACHABLE: '服务器未运行或无法访问'
	}
};

export const HTTP_CODE_TO_STRING: Record<string, string> = {
	401: ERROR_MESSAGES.HTTP.ACCESS_DENIED,
	403: ERROR_MESSAGES.HTTP.ACCESS_DENIED,
	500: ERROR_MESSAGES.HTTP.INTERNAL_ERROR,
	503: ERROR_MESSAGES.HTTP.TEMPORARILY_UNAVAILABLE
};

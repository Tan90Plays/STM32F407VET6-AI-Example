/* ???? ------------------------------------------------------------------*/
#include "stm32f4xx.h"
#include "delay.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "usart.h"
#include "ai_platform.h"
#include "network.h"
#include "network_data.h"
/* ???????? ----------------------------------------------------------------*/
static void ai_init(void);
static int ai_infer_360(const float* x360, float* y3);
static int parse_csv_floats(const char* s, float* dst, int n);
static int parse_csv_ints_norm01(const char* s, float* dst, int n);
static int uart_read_line(char* buf, int maxlen);

#define IN_LEN 360
#define OUT_LEN 3
#define RX_BUF_SZ USART_REC_LEN // 使用 usart.h 中定义的接收缓冲区大小

static ai_handle g_net = AI_HANDLE_NULL;
static uint8_t g_ai_ready = 0;

static AI_ALIGNED(32) uint8_t pool0[AI_NETWORK_DATA_ACTIVATION_1_SIZE];
static ai_handle data_activations0[] = { pool0 };

static ai_buffer* ai_input;
static ai_buffer* ai_output;

static float in_buf[IN_LEN];
static float out_buf[OUT_LEN];
static char rx_line[RX_BUF_SZ];

static void ai_init(void)
{
    ai_error err = ai_network_create_and_init(&g_net, data_activations0, NULL);
    if (err.type != AI_ERROR_NONE) {
        g_ai_ready = 0;
        return;
    }
    ai_input = ai_network_inputs_get(g_net, NULL);
    ai_output = ai_network_outputs_get(g_net, NULL);
    g_ai_ready = 1;
}

static int ai_infer_360(const float* x360, float* y3)
{
    if(!g_ai_ready) return 0;
    ai_input[0].data = AI_HANDLE_PTR((void*)x360);
    ai_output[0].data = AI_HANDLE_PTR((void*)y3);
    ai_i32 nb = ai_network_run(g_net, ai_input, ai_output);
    if(nb!=1) return 0;
    return 1;
}

static int parse_csv_floats(const char* s, float* dst, int n)
{
    int count = 0;                                  // 初始化已解析数字的计数器
    const char* p = s;                              // 使用指针p遍历输入字符串
    char* endp;                                     // 用于存储strtof解析停止的位置
    while (*p && count < n) {                       // 当字符串未结束且未达到期望数量n时循环
        float v = strtof(p, &endp);                 // 将当前位置的字符串转换为浮点数
        if (endp == p) break;                       // 如果转换未发生（指针未移动），跳出循环
        dst[count++] = v;                           // 将转换后的浮点数存入目标数组，计数加1
        if (*endp == 0) break;                      // 如果已到达字符串末尾，跳出循环
        p = endp + 1;                               // 将指针p移动到下一个数字的起始位置（跳过逗号）
    }
    return count;                                   // 返回实际解析成功的数字数量
}

static int parse_csv_ints_norm01(const char* s, float* dst, int n)
{
    static int tmp[IN_LEN];                         // 静态局部变量存储解析出的整数，防止栈溢出
    int count = 0;                                  // 初始化整数计数器
    const char* p = s;                              // 指针p指向输入字符串起始位置
    char* endp;                                     // 存储strtol解析停止的位置
    while (*p && count < n) {                       // 遍历字符串直至结束或达到n个数字
        long v = strtol(p, &endp, 10);              // 按10进制将字符串转换为长整型
        if (endp == p) break;                       // 若未成功转换任何数字，停止解析
        tmp[count++] = (int)v;                      // 存入临时数组并增加计数
        if (*endp == 0) break;                      // 若已处理至字符串结尾，停止循环
        p = endp + 1;                               // 移动指针跳过分隔符（逗号）
    }
    if (count != n) return 0;                       // 若解析数量不足n个，返回失败(0)
    int i, mn = tmp[0], mx = tmp[0];                // 初始化最大值和最小值
    for (i=1;i<n;i++){                              // 遍历数组查找实际的最大值和最小值
        if(tmp[i]<mn) mn=tmp[i];                    // 更新最小值
        if(tmp[i]>mx) mx=tmp[i];                    // 更新最大值
    }
    if (mx == mn) {                                 // 若所有数值相等（防止除以0）
        for (i=0;i<n;i++) dst[i]=0.5f;              // 统一设为中值0.5
        return 1;                                   // 返回成功
    }
    float scale = 1.0f/(float)(mx-mn);              // 计算归一化缩放比例
    for (i=0;i<n;i++) dst[i] = (tmp[i]-mn)*scale;   // 将原始数据线性映射到[0, 1]区间
    return 1;                                       // 返回解析并归一化成功(1)
}

static int uart_read_line(char* buf, int maxlen)
{
    if(USART_RX_STA & 0x8000){
        int len = USART_RX_STA & 0x3fff;
        if (len == 0) { // 过滤掉由 \r\n 产生的第二个空中断
            USART_RX_STA = 0;
            return 0;
        }
        if (len >= maxlen) len = maxlen-1;
        memcpy(buf, USART_RX_BUF, len);
        buf[len] = 0; 
        USART_RX_STA = 0;
        return len;
    }
    return 0;
}

int main(void)
{ 
    NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);
    delay_init();
    uart_init(115200);
    ai_init();

    printf("\r\nSTM32 AI ECG Classifier Ready\r\n");
    printf("Send data frame: $E,<360_ADC_ints_CSV>\\n or $F,<360_floats_CSV>\\n\r\n");
    printf("Example: $E,1023,1001,...,1100\\n\r\n");

    while(1)
    {
        int n = uart_read_line(rx_line, RX_BUF_SZ);
        if (n > 0) {
            if (rx_line[0] == '$') { // Check for start of frame marker
                if (rx_line[1]=='P') { // $P: ping
                    printf("PONG\r\n");
                    continue;
                }
                char* payload = strchr(rx_line, ',');
                if (payload && payload[1]) { // Ensure comma and data exist
                    printf("DBG: RX_LINE: %s\r\n", rx_line); // 打印原始接收行
                    printf("DBG: PAYLOAD: %s\r\n", payload + 1); // 打印解析后的 payload
                    payload++; // Move past the comma
                    int ok = 0;
                    if (rx_line[1]=='E') { // ADC integer data
                        ok = parse_csv_ints_norm01(payload, in_buf, IN_LEN);
                        printf("DBG: parse_ints_norm01 result: %d\r\n", ok);
                    } else if (rx_line[1]=='F') { // Normalized float data
                        int c = parse_csv_floats(payload, in_buf, IN_LEN);
                        ok = (c == IN_LEN);
                        printf("DBG: parse_floats result: %d, count: %d\r\n", ok, c);
                    } else {
                        printf("ERR:type\r\n"); // Unknown data type marker
                        ok = 0;
                    }
                    
                    if (ok) {
                        int runok = ai_infer_360(in_buf, out_buf);
                        if (runok) {
                            int pred = 0;
                            if (out_buf[1] > out_buf[pred]) pred = 1;
                            if (out_buf[2] > out_buf[pred]) pred = 2;
                            printf("P=[%.6f,%.6f,%.6f], pred=%d\r\n",
                                   out_buf[0], out_buf[1], out_buf[2], pred);
                        } else {
                            printf("ERR:run\r\n"); // AI inference failed
                        }
                    } else {
                        printf("ERR:parse\r\n"); // CSV parsing failed
                    }
                } else {
                    printf("ERR:format\r\n"); // Invalid frame format (e.g., no comma)
                }
            } else {
                printf("ERR:frame\r\n"); // Missing start of frame marker '$'
            }
        }
    }
    return 0;
}



// import React, { useEffect, useRef } from 'react';
// interface WebGLBackgroundProps {
//   className?: string;
//   isPaused?: boolean; // 新增
// }

// const WebGLBackground: React.FC<WebGLBackgroundProps> = ({ className, isPaused = false}) => {
//   const canvasRef = useRef<HTMLCanvasElement>(null);
//   const rendererRef = useRef<Renderer | null>(null);
//   const animationFrameId = useRef<number | null>(null); // 保存动画帧 ID
//   const isPausedRef = useRef<boolean>(isPaused);
//   useEffect(() => {
//     isPausedRef.current = isPaused;
//   }, [isPaused]);

//   useEffect(() => {
//     const canvas = canvasRef.current;
//     if (!canvas) return;

//     const dpr = Math.max(1, 0.5 * window.devicePixelRatio);
//     const renderer = new Renderer(canvas, dpr);
//     rendererRef.current = renderer;

//     const resize = () => {
//       const { innerWidth: width, innerHeight: height } = window;
//       canvas.width = width * dpr;
//       canvas.height = height * dpr;
//       if (renderer) {
//         renderer.updateScale(dpr);
//       }
//     };

//     const fragmentShaderSource = `#version 300 es
// precision highp float;
// out vec4 O;
// uniform float time;
// uniform vec2 resolution;
// #define FC gl_FragCoord.xy
// #define R resolution
// #define T time
// #define hue(a) (.6+.6*cos(6.3*(a)+vec3(0,83,21)))
// float rnd(float a) {
// 	vec2 p=fract(a*vec2(12.9898,78.233));	p+=dot(p,p*345.);
// 	return fract(p.x*p.y);
// }
// vec3 pattern(vec2 uv) {
// 	vec3 col=vec3(0);
// 	for (float i=.0; i++<20.;) {
// 		float a=rnd(i);
// 		vec2 n=vec2(a,fract(a*34.56)), p=sin(n*(T+7.)+T*.5);
// 		float d=dot(uv-p,uv-p);
// 		col+=.00125/d*hue(dot(uv,uv)+i*.125+T);
// 	}
// 	return col;
// }
// void main(void) {
// 	vec2 uv=(FC-.5*R)/min(R.x,R.y);
// 	vec3 col=vec3(0);
// 	float s=2.4,
// 	a=atan(uv.x,uv.y),
// 	b=length(uv);
// 	uv=vec2(a*5./6.28318,.05/tan(b)+T);
// 	uv=fract(uv)-.5;
// 	col+=pattern(uv*s);
// 	O=vec4(col,1);
// }`;

//     renderer.updateShader(fragmentShaderSource);
//     renderer.setup();
//     renderer.init();
//     resize();
//     window.addEventListener('resize', resize);

//     const renderLoop = (now: number) => {
//       if (!isPausedRef.current) {
//         renderer.render(now);
//       }
//       animationFrameId.current = requestAnimationFrame(renderLoop);
//     };
//     animationFrameId.current = requestAnimationFrame(renderLoop);

//     return () => {
//       window.removeEventListener('resize', resize);
//       if (animationFrameId.current !== null) {
//         cancelAnimationFrame(animationFrameId.current); // 取消动画帧
//       }
//       rendererRef.current = null;
//     };
//   }, []);

//   return (
//     <canvas
//   ref={canvasRef}
//   style={{
//     position: 'absolute',
//     top: 0,
//     left: 0,
//     width: '100%',
//     height: '100%',
//     display: 'block',
//   }}
// ></canvas>
//   );
// };

// export default WebGLBackground;

// class Renderer {
//   private vertexSrc = `#version 300 es
// precision highp float;
// in vec4 position;
// void main(){gl_Position=position;}`;
//   private fragmtSrc = `#version 300 es
// precision highp float;
// out vec4 O;
// uniform float time;
// uniform vec2 resolution;
// void main() {
//   vec2 uv = gl_FragCoord.xy / resolution;
//   O = vec4(uv, sin(time) * 0.5 + 0.5, 1.0);
// }`;
//   private vertices = new Float32Array([-1, 1, -1, -1, 1, 1, 1, -1]);
//   private gl: WebGL2RenderingContext;
//   private shaderSource: string;
//   private program!: WebGLProgram;
//   private buffer!: WebGLBuffer;
//   private canvas: HTMLCanvasElement;
//   private scale: number;

//   constructor(canvas: HTMLCanvasElement, scale: number) {
//     this.canvas = canvas;
//     this.scale = scale;
//     const gl = canvas.getContext('webgl2');
//     if (!gl) {
//       throw new Error('WebGL2 not supported');
//     }
//     this.gl = gl;
//     this.gl.viewport(0, 0, canvas.width * scale, canvas.height * scale);
//     this.shaderSource = this.fragmtSrc;
//   }

//   updateShader(source: string) {
//     this.reset();
//     this.shaderSource = source;
//     this.setup();
//     this.init();
//   }

//   updateScale(scale: number) {
//     this.scale = scale;
//     this.gl.viewport(0, 0, this.canvas.width * scale, this.canvas.height * scale);
//   }

//   compile(shader: WebGLShader, source: string) {
//     const gl = this.gl;
//     gl.shaderSource(shader, source);
//     gl.compileShader(shader);
//     if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
//       console.error(gl.getShaderInfoLog(shader));
//     }
//   }

//   test(source: string) {
//     const gl = this.gl;
//     const shader = gl.createShader(gl.FRAGMENT_SHADER)!;
//     gl.shaderSource(shader, source);
//     gl.compileShader(shader);
//     if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
//       const info = gl.getShaderInfoLog(shader);
//       console.error(info);
//       return info;
//     }
//     return null;
//   }

//   reset() {
//     const gl = this.gl;
//     if (this.program) {
//       gl.deleteProgram(this.program);
//     }
//   }

//   setup() {
//     const gl = this.gl;
//     const vs = gl.createShader(gl.VERTEX_SHADER)!;
//     const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
//     this.compile(vs, this.vertexSrc);
//     this.compile(fs, this.shaderSource);

//     this.program = gl.createProgram()!;
//     gl.attachShader(this.program, vs);
//     gl.attachShader(this.program, fs);
//     gl.linkProgram(this.program);

//     if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
//       console.error(gl.getProgramInfoLog(this.program));
//     }
//   }

//   init() {
//     const gl = this.gl;
//     this.buffer = gl.createBuffer()!;
//     gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
//     gl.bufferData(gl.ARRAY_BUFFER, this.vertices, gl.STATIC_DRAW);
//     const position = gl.getAttribLocation(this.program, 'position');
//     gl.enableVertexAttribArray(position);
//     gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
//   }

//   render(now: number) {
//     const gl = this.gl;
//     gl.clearColor(0, 0, 0, 1);
//     gl.clear(gl.COLOR_BUFFER_BIT);
//     gl.useProgram(this.program);
//     gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);

//     const resolutionLocation = gl.getUniformLocation(this.program, 'resolution');
//     const timeLocation = gl.getUniformLocation(this.program, 'time');

//     gl.uniform2f(resolutionLocation, this.canvas.width, this.canvas.height);
//     gl.uniform1f(timeLocation, now * 0.001);

//     gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
//   }
// }

import React, { useEffect, useRef, useState } from 'react';

interface WebGLBackgroundProps {
  className?: string;
  isPaused?: boolean; // 新增
  currentTime: number; // 新增
  songTitle:string;
}

const WebGLBackground: React.FC<WebGLBackgroundProps> = ({ className, isPaused = false, currentTime, songTitle }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<Renderer | null>(null);
  const animationFrameId = useRef<number | null>(null); // 保存动画帧 ID
  const isPausedRef = useRef<boolean>(isPaused);
  const [emotionData, setEmotionData] = useState<{ frameTime: number; valence: number; arousal: number }[]>([]);
  useEffect(() => {
    isPausedRef.current = isPaused;
  }, [isPaused]);
  useEffect(() => {
    const loadEmotionData = async (title: string) => {
      try {
        const response = await fetch(`/Emo-Proc/${title}.js`);
        if (!response.ok) {
          throw new Error(`Failed to fetch ${title}.js`);
        }
        const scriptText = await response.text();
        // 移除 export 关键字
        const cleanedScriptText = scriptText.replace('export default', '');
        const data = eval(cleanedScriptText + '; data;');
        setEmotionData(data);
      } catch (error) {
        console.error(`Error loading emotion data for ${title}:`, error);
        if (title !== 'Coastal') {
          // Try to load the default Coastal.js if the specified file doesn't exist
          loadEmotionData('Coastal');
          console.log(`Current songTitle: ${songTitle}`);
        }
      }
    };
  
    loadEmotionData(songTitle);
    console.log(`Current songTitle: ${songTitle}`);
  }, [songTitle]);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = Math.max(1, 0.5 * window.devicePixelRatio);
    const renderer = new Renderer(canvas, dpr);
    rendererRef.current = renderer;

    const resize = () => {
      const { innerWidth: width, innerHeight: height } = window;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      if (renderer) {
        renderer.updateScale(dpr);
      }
    };

    // 定义情绪数据
    

    // 插值函数
    const interpolateEmotion = (currentTime: number): { valence: number; arousal: number } => {
      if (emotionData.length === 0) {
        return { valence: 0, arousal: 0 };
      }
      if (currentTime <= emotionData[0].frameTime) {
        return { valence: emotionData[0].valence, arousal: emotionData[0].arousal };
      }
      if (currentTime >= emotionData[emotionData.length - 1].frameTime) {
        return {
          valence: emotionData[emotionData.length - 1].valence,
          arousal: emotionData[emotionData.length - 1].arousal,
        };
      }

      for (let i = 0; i < emotionData.length - 1; i++) {
        const currentData = emotionData[i];
        const nextData = emotionData[i + 1];

        if (currentTime >= currentData.frameTime && currentTime < nextData.frameTime) {
          const t = (currentTime - currentData.frameTime) / (nextData.frameTime - currentData.frameTime);
          const valence = currentData.valence + t * (nextData.valence - currentData.valence);
          const arousal = currentData.arousal + t * (nextData.arousal - currentData.arousal);
          return { valence, arousal };
        }
      }

      // 默认返回第一个数据点
      return { valence: emotionData[0].valence, arousal: emotionData[0].arousal };
    };

    // 计算当前的 valence 和 arousal
    const { valence, arousal } = interpolateEmotion(currentTime);

    // 片段着色器代码（已修改以使用 valence 和 arousal）
    const fragmentShaderSource = `#version 300 es
precision highp float;
out vec4 O;
uniform float time;
uniform vec2 resolution;
uniform float valence;  // 范围：-1.0 到 1.0
uniform float arousal;  // 范围：-1.0 到 1.0

#define FC gl_FragCoord.xy
#define R resolution
#define T time

// 定义颜色
#define red    vec3(1.0, 0.0, 0.0)    // 红色
#define yellow vec3(1.0, 1.0, 0.0)    // 黄色
#define blue   vec3(0.0, 0.0, 1.0)    // 蓝色
#define green  vec3(0.0, 1.0, 0.0)    // 绿色
#define purple vec3(0.5, 0.0, 1.0)    // 紫色
#define cyan   vec3(0.0, 1.0, 1.0)    // 青色
#define orange vec3(1.0, 0.5, 0.0)    // 橙色

float rnd(float a) {
    vec2 p = fract(a * vec2(12.9898,78.233));
    p += dot(p, p * 345.0);
    return fract(p.x * p.y);
}

// 计算颜色权重函数
void computeColorWeights(
    out float redWeight,
    out float yellowWeight,
    out float blueWeight,
    out float greenWeight,
    out float purpleWeight,
    out float cyanWeight,
    out float orangeWeight
) {
    // 初始化权重
    redWeight = 0.0;
    yellowWeight = 0.0;
    blueWeight = 0.0;
    greenWeight = 0.0;
    purpleWeight = 0.0;
    cyanWeight = 0.0;
    orangeWeight = 0.0;

    // 定义阈值
    float threshold = 0.5;
    float transitionWidth = 0.2; // 过渡宽度，用于 smoothstep

    // 计算 valence 和 arousal 超过阈值的部分
    // 使用 smoothstep 实现平滑过渡
    float valPos = smoothstep(threshold - transitionWidth, threshold + transitionWidth, valence);
    float valNeg = smoothstep(-threshold - transitionWidth, -threshold + transitionWidth, -valence);
    float aroPos = smoothstep(threshold - transitionWidth, threshold + transitionWidth, arousal);
    float aroNeg = smoothstep(-threshold - transitionWidth, -threshold + transitionWidth, -arousal);

    // 愤怒（高 valence，高 arousal）-> 红色和橙色
    float anger = valPos * aroPos;
    redWeight += anger * 0.5;        // 最多 50% 权重
    orangeWeight += anger * 0.3;     // 最多 30% 权重

    // 悲伤（低 valence，低 arousal）-> 蓝色和紫色
    float sadness = valNeg * aroNeg;
    blueWeight += sadness * 0.5;     // 最多 50% 权重
    purpleWeight += sadness * 0.3;   // 最多 30% 权重

    // 平静（valence 接近 0.0，低 arousal）-> 绿色和青色
    float calm = (1.0 - abs(valence)) * aroNeg;
    greenWeight += calm * 0.5;       // 最多 50% 权重
    cyanWeight += calm * 0.3;        // 最多 30% 权重

    // 快乐（高 valence，低 arousal）-> 黄色
    float happiness = valPos * aroNeg;
    yellowWeight += happiness * 0.5; // 最多 50% 权重

    // 确保权重总和为 1
    float totalWeight = redWeight + yellowWeight + blueWeight + greenWeight + purpleWeight + cyanWeight + orangeWeight;
    if (totalWeight > 0.0) {
        redWeight /= totalWeight;
        yellowWeight /= totalWeight;
        blueWeight /= totalWeight;
        greenWeight /= totalWeight;
        purpleWeight /= totalWeight;
        cyanWeight /= totalWeight;
        orangeWeight /= totalWeight;
    }
}

vec3 pattern(vec2 uv) {
    vec3 col = vec3(0.0);

    // 计算颜色权重
    float redWeight, yellowWeight, blueWeight, greenWeight, purpleWeight, cyanWeight, orangeWeight;
    computeColorWeights(redWeight, yellowWeight, blueWeight, greenWeight, purpleWeight, cyanWeight, orangeWeight);

    // 累积权重用于选择颜色
    float cumulativeWeights[7];
    cumulativeWeights[0] = redWeight;
    cumulativeWeights[1] = cumulativeWeights[0] + yellowWeight;
    cumulativeWeights[2] = cumulativeWeights[1] + blueWeight;
    cumulativeWeights[3] = cumulativeWeights[2] + greenWeight;
    cumulativeWeights[4] = cumulativeWeights[3] + purpleWeight;
    cumulativeWeights[5] = cumulativeWeights[4] + cyanWeight;
    cumulativeWeights[6] = cumulativeWeights[5] + orangeWeight;

    // 循环生成光束
    for (float i = 0.0; i < 20.0; i++) {
        float a = rnd(i);
        vec2 n = vec2(a, fract(a * 34.56));
        vec2 p = sin(n * (T + 7.0) + T * 0.5);
        float d = dot(uv - p, uv - p);

        // 生成用于选择颜色的随机数
        float randColor = rnd(i + T);

        // 根据权重选择颜色
        if (randColor < cumulativeWeights[0]) {
            col += 0.00125 / d * red;
        } else if (randColor < cumulativeWeights[1]) {
            col += 0.00125 / d * yellow;
        } else if (randColor < cumulativeWeights[2]) {
            col += 0.00125 / d * blue;
        } else if (randColor < cumulativeWeights[3]) {
            col += 0.00125 / d * green;
        } else if (randColor < cumulativeWeights[4]) {
            col += 0.00125 / d * purple;
        } else if (randColor < cumulativeWeights[5]) {
            col += 0.00125 / d * cyan;
        } else {
            col += 0.00125 / d * orange;
        }
    }

    return col;
}

void main(void) {
    vec2 uv = (FC - 0.5 * R) / min(R.x, R.y);
    vec3 col = vec3(0.0);
    float s = 2.4,
          a = atan(uv.x, uv.y),
          b = length(uv);
    uv = vec2(a * 5.0 / 6.28318, 0.05 / tan(b) + T);
    uv = fract(uv) - 0.5;
    col += pattern(uv * s);
    O = vec4(col, 1.0);
}
`;

    renderer.updateShader(fragmentShaderSource);
    renderer.setup();
    renderer.init();
    resize();
    window.addEventListener('resize', resize);

    // 渲染函数
    const renderNow = (now: number) => {
      renderer.setValence(valence);
      renderer.setArousal(arousal);
      renderer.render(now);
    };

    // 初始渲染
    renderNow(Date.now());

    // 设置渲染循环
    const renderLoop = (now: number) => {
      if (!isPausedRef.current) {
        renderNow(now);
      }
      animationFrameId.current = requestAnimationFrame(renderLoop);
    };
    animationFrameId.current = requestAnimationFrame(renderLoop);

    return () => {
      window.removeEventListener('resize', resize);
      if (animationFrameId.current !== null) {
        cancelAnimationFrame(animationFrameId.current); // 取消动画帧
      }
      if (rendererRef.current) {
        rendererRef.current.cleanup();
      }
      rendererRef.current = null;
    };
  }, [currentTime, emotionData]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        display: 'block',
      }}
    ></canvas>
  );
};

export default WebGLBackground;

class Renderer {
  private vertexSrc = `#version 300 es
precision highp float;
in vec4 position;
void main(){
  gl_Position = position;
}`;

  private fragmtSrc: string;
  private vertices = new Float32Array([-1, 1, -1, -1, 1, 1, 1, -1]);
  private gl: WebGL2RenderingContext;
  private program!: WebGLProgram;
  private buffer!: WebGLBuffer;
  private canvas: HTMLCanvasElement;
  private scale: number;
  private valence: number = 0.0;
  private arousal: number = 0.0;

  constructor(canvas: HTMLCanvasElement, scale: number) {
    this.canvas = canvas;
    this.scale = scale;
    const gl = canvas.getContext('webgl2');
    if (!gl) {
      throw new Error('WebGL2 not supported');
    }
    this.gl = gl;
    this.fragmtSrc = ''; // 将在外部设置
    this.gl.viewport(0, 0, canvas.width * scale, canvas.height * scale);
  }

  // 更新片段着色器源码
  updateShader(source: string) {
    this.reset();
    this.fragmtSrc = source;
    this.setup();
    this.init();
  }

  // 更新缩放
  updateScale(scale: number) {
    this.scale = scale;
    this.gl.viewport(0, 0, this.canvas.width * scale, this.canvas.height * scale);
  }

  // 编译着色器
  compile(shader: WebGLShader, source: string) {
    const gl = this.gl;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error('Shader compile error:', gl.getShaderInfoLog(shader));
    }
  }

  // 测试着色器
  test(source: string) {
    const gl = this.gl;
    const shader = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      console.error(info);
      return info;
    }
    gl.deleteShader(shader);
    return null;
  }

  // 重置着色器程序
  reset() {
    const gl = this.gl;
    if (this.program) {
      gl.deleteProgram(this.program);
    }
  }

  // 设置着色器程序
  setup() {
    const gl = this.gl;
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    this.compile(vs, this.vertexSrc);
    this.compile(fs, this.fragmtSrc);

    this.program = gl.createProgram()!;
    gl.attachShader(this.program, vs);
    gl.attachShader(this.program, fs);
    gl.linkProgram(this.program);

    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      console.error('Program link error:', gl.getProgramInfoLog(this.program));
    }
  }

  // 初始化缓冲区和属性
  init() {
    const gl = this.gl;
    this.buffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.vertices, gl.STATIC_DRAW);
    const position = gl.getAttribLocation(this.program, 'position');
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  }

  // 设置 valence 值
  setValence(val: number) {
    this.valence = val;
  }

  // 设置 arousal 值
  setArousal(arousal: number) {
    this.arousal = arousal;
  }

  // 渲染函数
  render(now: number) {
    const gl = this.gl;
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.program);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);

    // 设置 Uniform 变量
    const resolutionLocation = gl.getUniformLocation(this.program, 'resolution');
    const timeLocation = gl.getUniformLocation(this.program, 'time');
    const valenceLocation = gl.getUniformLocation(this.program, 'valence');
    const arousalLocation = gl.getUniformLocation(this.program, 'arousal');

    if (resolutionLocation) {
      gl.uniform2f(resolutionLocation, this.canvas.width, this.canvas.height);
    }
    if (timeLocation) {
      gl.uniform1f(timeLocation, now * 0.001);
    }
    if (valenceLocation) {
      gl.uniform1f(valenceLocation, this.valence);
    }
    if (arousalLocation) {
      gl.uniform1f(arousalLocation, this.arousal);
    }

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  // 清理资源
  cleanup() {
    const gl = this.gl;
    if (this.program) {
      gl.deleteProgram(this.program);
    }
    if (this.buffer) {
      gl.deleteBuffer(this.buffer);
    }
  }
}
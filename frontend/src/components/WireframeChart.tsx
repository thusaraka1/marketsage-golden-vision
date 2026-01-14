import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Line, Float } from "@react-three/drei";
import * as THREE from "three";

function ChartLine() {
  const lineRef = useRef<THREE.Group>(null);
  
  // Generate chart-like points
  const points = useMemo(() => {
    const pts: THREE.Vector3[] = [];
    const segments = 50;
    for (let i = 0; i <= segments; i++) {
      const x = (i / segments) * 8 - 4;
      const y = Math.sin(x * 0.8) * 1.5 + Math.sin(x * 2) * 0.5 + Math.random() * 0.2;
      pts.push(new THREE.Vector3(x, y, 0));
    }
    return pts;
  }, []);

  // Grid lines
  const gridLines = useMemo(() => {
    const lines: THREE.Vector3[][] = [];
    // Horizontal
    for (let i = -3; i <= 3; i++) {
      lines.push([new THREE.Vector3(-4, i, 0), new THREE.Vector3(4, i, 0)]);
    }
    // Vertical
    for (let i = -4; i <= 4; i++) {
      lines.push([new THREE.Vector3(i, -3, 0), new THREE.Vector3(i, 3, 0)]);
    }
    return lines;
  }, []);

  useFrame((state) => {
    if (lineRef.current) {
      lineRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.3) * 0.1;
      lineRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.2) * 0.05;
    }
  });

  return (
    <Float speed={2} rotationIntensity={0.2} floatIntensity={0.5}>
      <group ref={lineRef}>
        {/* Grid */}
        {gridLines.map((pts, i) => (
          <Line
            key={i}
            points={pts}
            color="#10b981"
            lineWidth={0.5}
            opacity={0.1}
            transparent
          />
        ))}
        
        {/* Main chart line */}
        <Line
          points={points}
          color="#10b981"
          lineWidth={3}
          opacity={0.9}
          transparent
        />
        
        {/* Glow line */}
        <Line
          points={points}
          color="#10b981"
          lineWidth={8}
          opacity={0.15}
          transparent
        />
      </group>
    </Float>
  );
}

const WireframeChart = () => {
  return (
    <div className="w-full h-full min-h-[400px]">
      <Canvas
        camera={{ position: [0, 0, 8], fov: 50 }}
        style={{ background: "transparent" }}
      >
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={0.5} />
        <ChartLine />
      </Canvas>
    </div>
  );
};

export default WireframeChart;

"use client";

import { useEffect, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { LoadingManager, Mesh, Material } from "three";
import URDFLoader, { URDFRobot } from "urdf-loader";
import { useTime } from "@/context/time-context";
import { fetchRobotMotion, type RobotMotion } from "@/utils/annotationsClient";
import { robotFrameAt } from "@/utils/robotMotion";

function disposeRobot(robot: URDFRobot) {
  robot.traverse((object) => {
    if (object instanceof Mesh) {
      object.geometry.dispose();
      const materials = Array.isArray(object.material)
        ? object.material
        : [object.material];
      materials.forEach((material: Material) => material.dispose());
    }
  });
}

function Motion({ robot, motion }: { robot: URDFRobot; motion: RobotMotion }) {
  const { currentTime, subscribe } = useTime();
  const invalidate = useThree((state) => state.invalidate);
  const time = useRef(currentTime);
  useEffect(() => {
    time.current = currentTime;
    invalidate();
  }, [currentTime, invalidate]);
  useEffect(
    () =>
      subscribe((value) => {
        time.current = value;
        invalidate();
      }),
    [subscribe, invalidate],
  );
  useFrame(() => {
    const index = robotFrameAt(motion.timestamps, time.current);
    motion.joint_names.forEach((name, joint) =>
      robot.setJointValue(name, motion.positions[index][joint]),
    );
    const q = motion.root_orientations?.[index];
    if (q) robot.quaternion.set(q[1], q[2], q[3], q[0]).normalize();
  });
  // URDF uses Z up. Rotate the complete world into Three's Y-up coordinates.
  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      <primitive object={robot} dispose={null} />
    </group>
  );
}

export default function G1MotionReplay({
  repoId,
  episodeId,
}: {
  repoId: string;
  episodeId: number;
}) {
  const [motion, setMotion] = useState<RobotMotion | null>(null);
  const [robot, setRobot] = useState<URDFRobot | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    let loaded: URDFRobot | undefined;
    setMotion(null);
    setRobot(null);
    setError(null);
    const manager = new LoadingManager();
    const fail = () => {
      if (active) setError("Could not load the local G1 model or its meshes.");
    };
    manager.onError = fail;
    manager.onLoad = () => {
      if (loaded && active) setRobot(loaded);
      else if (loaded) disposeRobot(loaded);
    };
    new URDFLoader(manager).load(
      "/robots/g1/g1_29dof_with_hand.urdf",
      (value) => {
        loaded = value;
        // Recorded measurements must not be clamped to nominal URDF limits.
        Object.values(value.joints).forEach((joint) => {
          joint.ignoreLimits = true;
        });
      },
      undefined,
      fail,
    );
    fetchRobotMotion(episodeId, { repoId }, controller.signal)
      .then((value) => {
        if (active) setMotion(value);
      })
      .catch((reason) => {
        if (active) setError(String(reason.message || reason));
      });
    return () => {
      active = false;
      controller.abort();
      if (loaded) disposeRobot(loaded);
    };
  }, [repoId, episodeId]);
  const unknown =
    robot && motion
      ? motion.joint_names.filter((name) => !robot.joints[name])
      : [];
  const problem =
    error ||
    (unknown.length
      ? `Joint names do not match the G1 URDF: ${unknown.join(", ")}`
      : null);
  return (
    <section className="g1-replay panel" aria-label="G1 motion replay">
      <div className="px-4 pt-3 text-sm font-medium">G1 motion replay</div>
      <p className="px-4 text-xs text-slate-400">
        Measured joints
        {motion?.root_orientations ? " and root orientation" : ""}. Root
        position not recorded; fixed-base replay.
      </p>
      <div className="g1-replay-canvas">
        {problem ? (
          <p role="alert" className="p-4 text-amber-300">
            {problem}
          </p>
        ) : robot && motion ? (
          <Canvas
            frameloop="demand"
            camera={{ position: [2, 1.1, 2], fov: 40 }}
            dpr={[1, 1.5]}
            fallback={
              <p>
                WebGL is unavailable. Enable browser hardware acceleration to
                view the robot.
              </p>
            }
          >
            <ambientLight intensity={1.5} />
            <directionalLight position={[3, 5, 4]} intensity={3} />
            <Motion robot={robot} motion={motion} />
            <gridHelper
              args={[4, 20, "#475569", "#263244"]}
              position={[0, -0.78, 0]}
            />
            <OrbitControls
              target={[0, -0.05, 0]}
              minDistance={0.5}
              maxDistance={8}
            />
          </Canvas>
        ) : (
          <p role="status" className="p-4">
            Loading recorded motion and G1 model…
          </p>
        )}
      </div>
      <p className="px-4 pb-3 text-xs text-slate-400">
        Drag to orbit · Scroll to zoom · Use the video timeline to play or seek
      </p>
    </section>
  );
}

import { useRef, useEffect, useState } from "react";

export default function Canvas({ width, height, canvasRef, ctxRef, isMobile }) {
  const [isDrawing, setIsDrawing] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width = width * 2;
    canvas.height = height * 2;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext("2d");
    ctx.scale(2, 2);
    ctx.lineCap = "round";
    ctx.strokeStyle = "white";
    ctx.lineWidth = isMobile ? 30 : 40;
    ctxRef.current = ctx;
  }, []);

  const startDrawing = (e) => {
    if (isMobile && e.touches.length !== 1) return;
    const { clientX, clientY } = isMobile ? e.touches[0] : e;
    const { offsetX, offsetY } = getOffset(clientX, clientY);
    ctxRef.current.beginPath();
    ctxRef.current.moveTo(offsetX, offsetY);
    setIsDrawing(true);
  };

  const finishDrawing = () => {
    ctxRef.current.closePath();
    setIsDrawing(false);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    if (isMobile && e.touches.length !== 1) return;
    const { clientX, clientY } = isMobile ? e.touches[0] : e;
    const { offsetX, offsetY } = getOffset(clientX, clientY);
    ctxRef.current.lineTo(offsetX, offsetY);
    ctxRef.current.stroke();
  };

  const getOffset = (clientX, clientY) => {
    const canvas = canvasRef.current;
    const { top, left } = canvas.getBoundingClientRect();
    const offsetX = clientX - left;
    const offsetY = clientY - top;
    return { offsetX, offsetY };
  };

  return (
    <>
      <canvas
        className="bg-black border-2 border-white touch-none"
        id="canvas"
        ref={canvasRef}
        onMouseDown={startDrawing}
        onMouseUp={finishDrawing}
        onMouseMove={draw}
        onTouchStart={isMobile ? startDrawing : undefined}
        onTouchEnd={isMobile ? finishDrawing : undefined}
        onTouchMove={isMobile ? draw : undefined}
      />
    </>
  );
}

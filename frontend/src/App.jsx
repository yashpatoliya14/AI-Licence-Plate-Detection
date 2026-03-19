import React, { useState, useRef } from "react";
// eslint-disable-next-line no-unused-vars
import { motion, AnimatePresence } from "framer-motion";
import { UploadCloud, ArrowLeft, Camera, ShieldCheck, Zap, Activity, ScanLine, X, Search } from "lucide-react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

// Utility for Tailwind classes
function cn(...inputs) {
  return twMerge(clsx(inputs));
}

// Custom Button mimicking Shadcn
const Button = React.forwardRef(({ className, variant = "default", size = "default", ...props }, ref) => {
  const variants = {
    default: "bg-primary text-primary-foreground hover:bg-primary/90",
    destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
    outline: "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
    secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
    ghost: "hover:bg-accent hover:text-accent-foreground",
    link: "text-primary underline-offset-4 hover:underline",
  };
  const sizes = {
    default: "h-10 px-4 py-2",
    sm: "h-9 rounded-md px-3",
    lg: "h-11 rounded-md px-8",
    icon: "h-10 w-10",
  };

  return (
    <button
      ref={ref}
      className={cn(
        "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
        variants[variant],
        sizes[size],
        className
      )}
      {...props}
    />
  );
});
Button.displayName = "Button";

// Custom Card mimicking Shadcn
const Card = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("rounded-xl border bg-card text-card-foreground shadow focus-visible:outline-none", className)} {...props} />
));
Card.displayName = "Card";

export default function App() {
  const [view, setView] = useState("home"); // 'home' or 'scanner'
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);

  const fileInputRef = useRef(null);

  const startAnalysis = () => {
    setView("scanner");
    resetScanner();
  };

  const resetScanner = () => {
    setSelectedImage(null);
    setImagePreviewUrl(null);
    setResults(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(file);
      setImagePreviewUrl(URL.createObjectURL(file));
      setResults(null);
    }
  };

  const triggerFileInput = () => {
    if (!isProcessing) {
      fileInputRef.current?.click();
    }
  };

  const handleUpload = async () => {
    if (!selectedImage) return;

    setIsProcessing(true);
    setResults(null);

    const formData = new FormData();
    formData.append("file", selectedImage);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        if (data.detections && data.detections.length > 0) {
          const firstDetection = data.detections[0];
          setResults({
            text: firstDetection.text || "No text detected",
            confidence: firstDetection.confidence ? `${(firstDetection.confidence * 100).toFixed(1)}%` : "N/A",
            plate_crop: firstDetection.image,
          });
        } else {
          setResults({
            text: "No plates found",
            confidence: "N/A",
            plate_crop: null,
          });
        }
      } else {
        alert("Error during detection. The backend might not be responding properly.");
      }
    } catch (error) {
      console.error("Error connecting to backend:", error);
      alert("Failed to connect to backend server. Make sure the FastAPI server is running.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground selection:bg-primary/30 font-sans overflow-x-hidden relative">
      <div className="absolute top-0 w-full h-full -z-10 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/20 via-background to-background" />

      <header className="fixed top-0 w-full border-b backdrop-blur-md bg-background/80 z-50">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2 cursor-pointer" onClick={() => setView("home")}>
            <div className="p-2 bg-primary/20 rounded-lg text-primary">
              <Camera size={20} />
            </div>
            <span className="font-bold text-lg tracking-tight">ALPR System</span>
          </div>
          {view === "scanner" && (
            <Button variant="ghost" size="sm" onClick={() => setView("home")} className="gap-2">
              <ArrowLeft size={16} /> Home
            </Button>
          )}
        </div>
      </header>

      <main className="pt-24 pb-16 min-h-screen px-6 max-w-6xl mx-auto flex flex-col justify-center">
        <AnimatePresence mode="wait">
          {view === "home" && (
            <motion.div
              key="home"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4 }}
              className="w-full flex flex-col items-center text-center mt-12 md:mt-24"
            >
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.1, duration: 0.5 }}
                className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-primary/30 bg-primary/10 text-primary text-sm font-medium mb-8"
              >
                <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                AI-Powered Plate Analysis
              </motion.div>

              <motion.h1 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2, duration: 0.5 }}
                className="text-5xl md:text-7xl font-extrabold tracking-tight mb-6"
              >
                Detecting <br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-purple-400">
                  License Plates
                </span>
              </motion.h1>

              <motion.p 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3, duration: 0.5 }}
                className="text-muted-foreground text-lg md:text-xl max-w-2xl mb-10 leading-relaxed"
              >
                Uses a pretrained deep learning model (YOLOv8) to accurately locate
                and extract text from vehicle license plates in seconds. Robust, fast, and accurate.
              </motion.p>

              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.4, duration: 0.5 }}
              >
                <Button size="lg" className="h-14 px-8 text-lg gap-2 shadow-lg shadow-primary/25 rounded-full" onClick={startAnalysis}>
                  Start Analysis <ScanLine size={20} />
                </Button>
              </motion.div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-24 mb-12 w-full text-left">
                {[
                  { title: "Instant Processing", desc: "Get your plate detection results almost immediately.", icon: Zap, color: "text-amber-400", bg: "bg-amber-400/10" },
                  { title: "High Accuracy", desc: "State-of-the-art YOLOv8 object detection combined with EasyOCR.", icon: Activity, color: "text-blue-400", bg: "bg-blue-400/10" },
                  { title: "Private & Secure", desc: "Imagery is processed directly without permanent storage.", icon: ShieldCheck, color: "text-emerald-400", bg: "bg-emerald-400/10" },
                ].map((feature, i) => (
                  <motion.div 
                    key={i}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 + i * 0.1, duration: 0.5 }}
                  >
                    <Card className="p-6 h-full bg-card/50 backdrop-blur-sm border-white/5 hover:border-primary/50 transition-colors">
                      <div className={cn("w-12 h-12 rounded-xl flex items-center justify-center mb-4", feature.bg)}>
                        <feature.icon className={feature.color} size={24} />
                      </div>
                      <h3 className="font-semibold text-xl mb-2">{feature.title}</h3>
                      <p className="text-muted-foreground">{feature.desc}</p>
                    </Card>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}

          {view === "scanner" && (
            <motion.div
              key="scanner"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.4 }}
              className="w-full max-w-3xl mx-auto"
            >
              <div className="mb-8">
                <h2 className="text-3xl font-bold tracking-tight mb-2">New Detection</h2>
                <p className="text-muted-foreground">Upload an image of a vehicle to analyze its license plate.</p>
              </div>

              <Card className="p-1 md:p-2 bg-card/50 backdrop-blur-xl border-white/10 shadow-2xl relative overflow-hidden">
                <div className="p-6 md:p-8">
                  {!selectedImage ? (
                    <div
                      onClick={triggerFileInput}
                      className="border-2 border-dashed border-border hover:border-primary/50 rounded-xl p-12 flex flex-col items-center justify-center text-center cursor-pointer transition-all bg-secondary/20 hover:bg-secondary/40 group"
                    >
                      <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                        <UploadCloud className="text-primary w-8 h-8" />
                      </div>
                      <h3 className="text-lg font-medium mb-1">Click to upload or drag and drop</h3>
                      <p className="text-sm text-muted-foreground">SVG, PNG, JPG or GIF (max. 800x400px)</p>
                    </div>
                  ) : (
                    <div className="space-y-6">
                      <div className="relative rounded-xl overflow-hidden border bg-black/50 aspect-video flex items-center justify-center">
                        <img 
                          src={imagePreviewUrl} 
                          alt="Preview" 
                          className={cn(
                            "max-w-full max-h-full object-contain transition-all duration-500", 
                            isProcessing && "opacity-50 blur-[2px]"
                          )} 
                        />

                        {isProcessing && (
                          <>
                            <div className="absolute inset-0 z-10 animate-scanner pointer-events-none" />
                            <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-background/20 backdrop-blur-sm">
                              <div className="w-12 h-12 border-4 border-primary/30 border-t-primary rounded-full animate-spin mb-4" />
                              <p className="font-semibold tracking-widest text-primary animate-pulse">INTERROGATING IMAGE...</p>
                            </div>
                          </>
                        )}
                        
                        {!isProcessing && (
                           <Button 
                             variant="destructive" 
                             size="icon" 
                             className="absolute top-2 right-2 rounded-full h-8 w-8 opacity-80 hover:opacity-100"
                             onClick={(e) => { e.stopPropagation(); resetScanner(); }}
                           >
                             <X size={16} />
                           </Button>
                        )}
                      </div>

                      <AnimatePresence>
                        {results && (
                          <motion.div 
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: "auto" }}
                            className="overflow-hidden"
                          >
                            <Card className="bg-primary/5 border-primary/20 p-6">
                              <h3 className="font-semibold text-primary mb-4 flex items-center gap-2">
                                <Search size={18} /> Analysis Results
                              </h3>
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div>
                                  <p className="text-sm text-muted-foreground mb-1">Detected Plate Number</p>
                                  <p className="text-3xl font-bold font-mono tracking-wider">{results.text}</p>
                                </div>
                                <div>
                                  <p className="text-sm text-muted-foreground mb-1">Confidence Score</p>
                                  <div className="flex items-center gap-3">
                                    <p className="text-xl font-semibold">{results.confidence}</p>
                                    <div className="flex-1 h-2 bg-secondary rounded-full overflow-hidden">
                                      <div 
                                        className="h-full bg-primary" 
                                        style={{ width: results.confidence !== 'N/A' ? results.confidence : '0%' }}
                                      />
                                    </div>
                                  </div>
                                </div>
                                
                                <div className="md:col-span-2 mt-2">
                                  <p className="text-sm text-muted-foreground mb-3">Extracted Plate Region</p>
                                  {results.plate_crop ? (
                                    <div className="p-2 border rounded-lg bg-black/40 inline-block">
                                      <img 
                                        src={`data:image/jpeg;base64,${results.plate_crop}`} 
                                        alt="Plate Crop" 
                                        className="h-16 md:h-20 object-contain rounded" 
                                      />
                                    </div>
                                  ) : (
                                    <p className="text-sm italic text-muted-foreground bg-secondary/50 p-3 rounded-md border border-dashed">No plate crop available</p>
                                  )}
                                </div>
                              </div>
                            </Card>
                          </motion.div>
                        )}
                      </AnimatePresence>

                      <div className="flex justify-end gap-3 pt-4 border-t">
                        <Button variant="outline" onClick={resetScanner} disabled={isProcessing}>
                          Reset Image
                        </Button>
                        <Button 
                          onClick={handleUpload} 
                          disabled={isProcessing}
                          className="min-w-[140px]"
                        >
                          {isProcessing ? "Processing..." : (
                            <>
                              Run Analysis <Zap size={16} className="ml-2" />
                            </>
                          )}
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
              </Card>

              <input
                type="file"
                className="hidden"
                onChange={handleImageChange}
                ref={fileInputRef}
                accept="image/*"
              />
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

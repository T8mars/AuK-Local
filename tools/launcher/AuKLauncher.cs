using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;

[assembly: AssemblyTitle("AuK Local Launcher")]
[assembly: AssemblyDescription("Portable launcher for the AuK Local speech workstation")]
[assembly: AssemblyCompany("T8star-Aix")]
[assembly: AssemblyProduct("AuK Local")]
[assembly: AssemblyCopyright("Copyright © T8star-Aix 2026")]
[assembly: AssemblyVersion("0.1.1.0")]
[assembly: AssemblyFileVersion("0.1.1.0")]

internal static class AuKLauncher
{
    private static int Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;
        Console.Title = "AuK Local";

        string executable = Assembly.GetExecutingAssembly().Location;
        string packageRoot = Path.GetDirectoryName(executable);
        if (String.IsNullOrEmpty(packageRoot))
        {
            return Fail("无法确定 AuK Local 整合包目录。");
        }

        string python = Path.Combine(packageRoot, "runtime", "python.exe");
        string script = Path.Combine(packageRoot, "scripts", "Start-AuK.cmd");
        string source = Path.Combine(packageRoot, "src", "auk_local");

        if (args.Length == 1 && args[0] == "--print-root")
        {
            Console.WriteLine(packageRoot);
            return 0;
        }

        if (!File.Exists(python))
        {
            return Fail("整合包不完整，缺少 runtime\\python.exe。\n请完整解压后再启动，不要单独复制 EXE。");
        }
        if (!File.Exists(script))
        {
            return Fail("整合包不完整，缺少 scripts\\Start-AuK.cmd。");
        }
        if (!Directory.Exists(source))
        {
            return Fail("整合包不完整，缺少 src\\auk_local。\n请重新解压完整整合包。");
        }

        if (args.Length == 1 && args[0] == "--verify-package")
        {
            Console.WriteLine("AuK Local launcher: PASS");
            Console.WriteLine("Package root: " + packageRoot);
            return 0;
        }

        try
        {
            ProcessStartInfo info = new ProcessStartInfo();
            info.FileName = Environment.GetEnvironmentVariable("COMSPEC") ?? "cmd.exe";
            info.Arguments = "/d /c call scripts\\Start-AuK.cmd";
            info.WorkingDirectory = packageRoot;
            info.UseShellExecute = false;
            info.CreateNoWindow = false;

            using (Process process = Process.Start(info))
            {
                if (process == null)
                {
                    return Fail("无法创建 AuK 启动进程。");
                }
                process.WaitForExit();
                return process.ExitCode;
            }
        }
        catch (Exception error)
        {
            return Fail("启动 AuK 失败：" + error.Message);
        }
    }

    private static int Fail(string message)
    {
        Console.Error.WriteLine();
        Console.Error.WriteLine("[AuK 启动器错误] " + message);
        Console.Error.WriteLine();
        if (!Console.IsInputRedirected)
        {
            Console.Error.WriteLine("按任意键关闭此窗口...");
            Console.ReadKey(true);
        }
        return 1;
    }
}
